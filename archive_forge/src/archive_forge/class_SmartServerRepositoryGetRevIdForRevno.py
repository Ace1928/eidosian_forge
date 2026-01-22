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
class SmartServerRepositoryGetRevIdForRevno(SmartServerRepositoryReadLocked):

    def do_readlocked_repository_request(self, repository, revno, known_pair):
        """Find the revid for a given revno, given a known revno/revid pair.

        New in 1.17.
        """
        try:
            found_flag, result = repository.get_rev_id_for_revno(revno, known_pair)
        except errors.NoSuchRevision as err:
            if err.revision != known_pair[1]:
                raise AssertionError('get_rev_id_for_revno raised RevisionNotPresent for non-initial revision: ' + err.revision)
            return FailedSmartServerResponse((b'nosuchrevision', err.revision))
        except errors.RevnoOutOfBounds as e:
            return FailedSmartServerResponse((b'revno-outofbounds', e.revno, e.minimum, e.maximum))
        if found_flag:
            return SuccessfulSmartServerResponse((b'ok', result))
        else:
            earliest_revno, earliest_revid = result
            return SuccessfulSmartServerResponse((b'history-incomplete', earliest_revno, earliest_revid))