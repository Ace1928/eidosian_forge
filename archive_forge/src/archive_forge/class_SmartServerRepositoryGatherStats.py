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
class SmartServerRepositoryGatherStats(SmartServerRepositoryRequest):

    def do_repository_request(self, repository, revid, committers):
        """Return the result of repository.gather_stats().

        :param repository: The repository to query in.
        :param revid: utf8 encoded rev id or an empty string to indicate None
        :param committers: 'yes' or 'no'.

        :return: A SmartServerResponse (b'ok',), a encoded body looking like
              committers: 1
              firstrev: 1234.230 0
              latestrev: 345.700 3600
              revisions: 2

              But containing only fields returned by the gather_stats() call
        """
        if revid == b'':
            decoded_revision_id = None
        else:
            decoded_revision_id = revid
        if committers == b'yes':
            decoded_committers = True
        else:
            decoded_committers = None
        try:
            stats = repository.gather_stats(decoded_revision_id, decoded_committers)
        except errors.NoSuchRevision:
            return FailedSmartServerResponse((b'nosuchrevision', revid))
        body = b''
        if 'committers' in stats:
            body += b'committers: %d\n' % stats['committers']
        if 'firstrev' in stats:
            body += b'firstrev: %.3f %d\n' % stats['firstrev']
        if 'latestrev' in stats:
            body += b'latestrev: %.3f %d\n' % stats['latestrev']
        if 'revisions' in stats:
            body += b'revisions: %d\n' % stats['revisions']
        if 'size' in stats:
            body += b'size: %d\n' % stats['size']
        return SuccessfulSmartServerResponse((b'ok',), body)