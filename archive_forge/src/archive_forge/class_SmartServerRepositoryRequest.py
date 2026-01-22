import bz2
import itertools
import os
import queue
import sys
import tempfile
import threading
import zlib
import fastbencode as bencode
from ... import errors, estimate_compressed_size, osutils
from ... import revision as _mod_revision
from ... import trace, ui
from ...repository import _strip_NULL_ghosts, network_format_registry
from .. import inventory as _mod_inventory
from .. import inventory_delta, pack, vf_search
from ..bzrdir import BzrDir
from ..versionedfile import (ChunkedContentFactory, NetworkRecordStream,
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerRepositoryRequest(SmartServerRequest):
    """Common base class for Repository requests."""

    def do(self, path, *args):
        """Execute a repository request.

        All Repository requests take a path to the repository as their first
        argument.  The repository must be at the exact path given by the
        client - no searching is done.

        The actual logic is delegated to self.do_repository_request.

        :param client_path: The path for the repository as received from the
            client.
        :return: A SmartServerResponse from self.do_repository_request().
        """
        transport = self.transport_from_client_path(path)
        bzrdir = BzrDir.open_from_transport(transport)
        self._repository = bzrdir.open_repository()
        return self.do_repository_request(self._repository, *args)

    def do_repository_request(self, repository, *args):
        """Override to provide an implementation for a verb."""
        return None

    def recreate_search(self, repository, search_bytes, discard_excess=False):
        """Recreate a search from its serialised form.

        :param discard_excess: If True, and the search refers to data we don't
            have, just silently accept that fact - the verb calling
            recreate_search trusts that clients will look for missing things
            they expected and get it from elsewhere.
        """
        if search_bytes == b'everything':
            return (vf_search.EverythingResult(repository), None)
        lines = search_bytes.split(b'\n')
        if lines[0] == b'ancestry-of':
            heads = lines[1:]
            search_result = vf_search.PendingAncestryResult(heads, repository)
            return (search_result, None)
        elif lines[0] == b'search':
            return self.recreate_search_from_recipe(repository, lines[1:], discard_excess=discard_excess)
        else:
            return (None, FailedSmartServerResponse((b'BadSearch',)))

    def recreate_search_from_recipe(self, repository, lines, discard_excess=False):
        """Recreate a specific revision search (vs a from-tip search).

        :param discard_excess: If True, and the search refers to data we don't
            have, just silently accept that fact - the verb calling
            recreate_search trusts that clients will look for missing things
            they expected and get it from elsewhere.
        """
        start_keys = set(lines[0].split(b' '))
        exclude_keys = set(lines[1].split(b' '))
        revision_count = int(lines[2].decode('ascii'))
        with repository.lock_read():
            search = repository.get_graph()._make_breadth_first_searcher(start_keys)
            while True:
                try:
                    next_revs = next(search)
                except StopIteration:
                    break
                search.stop_searching_any(exclude_keys.intersection(next_revs))
            started_keys, excludes, included_keys = search.get_state()
            if not discard_excess and len(included_keys) != revision_count:
                return (None, FailedSmartServerResponse((b'NoSuchRevision',)))
            search_result = vf_search.SearchResult(started_keys, excludes, len(included_keys), included_keys)
            return (search_result, None)