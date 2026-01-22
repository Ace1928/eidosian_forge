import logging
import os
import select
import socket
import subprocess
import sys
from contextlib import closing
from io import BufferedReader, BytesIO
from typing import (
from urllib.parse import quote as urlquote
from urllib.parse import unquote as urlunquote
from urllib.parse import urljoin, urlparse, urlunparse, urlunsplit
import dulwich
from .config import Config, apply_instead_of, get_xdg_config_home_path
from .errors import GitProtocolError, NotGitRepository, SendPackError
from .pack import (
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, _import_remote_refs, read_info_refs
from .repo import Repo
class LocalGitClient(GitClient):
    """Git Client that just uses a local Repo."""

    def __init__(self, thin_packs=True, report_activity=None, config: Optional[Config]=None) -> None:
        """Create a new LocalGitClient instance.

        Args:
          thin_packs: Whether or not thin packs should be retrieved
          report_activity: Optional callback for reporting transport
            activity.
        """
        self._report_activity = report_activity

    def get_url(self, path):
        return urlunsplit(('file', '', path, '', ''))

    @classmethod
    def from_parsedurl(cls, parsedurl, **kwargs):
        return cls(**kwargs)

    @classmethod
    def _open_repo(cls, path):
        if not isinstance(path, str):
            path = os.fsdecode(path)
        return closing(Repo(path))

    def send_pack(self, path, update_refs, generate_pack_data, progress=None):
        """Upload a pack to a remote repository.

        Args:
          path: Repository path (as bytestring)
          update_refs: Function to determine changes to remote refs.
            Receive dict with existing remote refs, returns dict with
            changed refs (name -> sha, where sha=ZERO_SHA for deletions)
            with number of items and pack data to upload.
          progress: Optional progress function

        Returns:
          SendPackResult

        Raises:
          SendPackError: if server rejects the pack data

        """
        if not progress:

            def progress(x):
                pass
        with self._open_repo(path) as target:
            old_refs = target.get_refs()
            new_refs = update_refs(dict(old_refs))
            have = [sha1 for sha1 in old_refs.values() if sha1 != ZERO_SHA]
            want = []
            for refname, new_sha1 in new_refs.items():
                if new_sha1 not in have and new_sha1 not in want and (new_sha1 != ZERO_SHA):
                    want.append(new_sha1)
            if not want and set(new_refs.items()).issubset(set(old_refs.items())):
                return SendPackResult(new_refs, ref_status={})
            target.object_store.add_pack_data(*generate_pack_data(have, want, ofs_delta=True))
            ref_status = {}
            for refname, new_sha1 in new_refs.items():
                old_sha1 = old_refs.get(refname, ZERO_SHA)
                if new_sha1 != ZERO_SHA:
                    if not target.refs.set_if_equals(refname, old_sha1, new_sha1):
                        msg = f'unable to set {refname} to {new_sha1}'
                        progress(msg)
                        ref_status[refname] = msg
                elif not target.refs.remove_if_equals(refname, old_sha1):
                    progress('unable to remove %s' % refname)
                    ref_status[refname] = 'unable to remove'
        return SendPackResult(new_refs, ref_status=ref_status)

    def fetch(self, path, target, determine_wants=None, progress=None, depth=None):
        """Fetch into a target repository.

        Args:
          path: Path to fetch from (as bytestring)
          target: Target repository to fetch into
          determine_wants: Optional function determine what refs
            to fetch. Receives dictionary of name->sha, should return
            list of shas to fetch. Defaults to all shas.
          progress: Optional progress function
          depth: Shallow fetch depth

        Returns:
          FetchPackResult object

        """
        with self._open_repo(path) as r:
            refs = r.fetch(target, determine_wants=determine_wants, progress=progress, depth=depth)
            return FetchPackResult(refs, r.refs.get_symrefs(), agent_string())

    def fetch_pack(self, path, determine_wants, graph_walker, pack_data, progress=None, depth=None) -> FetchPackResult:
        """Retrieve a pack from a git smart server.

        Args:
          path: Remote path to fetch from
          determine_wants: Function determine what refs
            to fetch. Receives dictionary of name->sha, should return
            list of shas to fetch.
          graph_walker: Object with next() and ack().
          pack_data: Callback called for each bit of data in the pack
          progress: Callback for progress reports (strings)
          depth: Shallow fetch depth

        Returns:
          FetchPackResult object

        """
        with self._open_repo(path) as r:
            missing_objects = r.find_missing_objects(determine_wants, graph_walker, progress=progress, depth=depth)
            other_haves = missing_objects.get_remote_has()
            object_ids = list(missing_objects)
            symrefs = r.refs.get_symrefs()
            agent = agent_string()
            if object_ids is None:
                return FetchPackResult(None, symrefs, agent)
            write_pack_from_container(pack_data, r.object_store, object_ids, other_haves=other_haves)
            return FetchPackResult(r.get_refs(), symrefs, agent)

    def get_refs(self, path):
        """Retrieve the current refs from a git smart server."""
        with self._open_repo(path) as target:
            return target.get_refs()