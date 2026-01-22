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
class SmartServerRepositoryInsertStream_1_19(SmartServerRepositoryInsertStreamLocked):
    """Insert a record stream from a RemoteSink into a repository.

    Same as SmartServerRepositoryInsertStreamLocked, except:
     - the lock token argument is optional
     - servers that implement this verb accept 'inventory-delta' records in the
       stream.

    New in 1.19.
    """

    def do_repository_request(self, repository, resume_tokens, lock_token=None):
        """StreamSink.insert_stream for a remote repository."""
        SmartServerRepositoryInsertStreamLocked.do_repository_request(self, repository, resume_tokens, lock_token)