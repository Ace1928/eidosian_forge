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
class SmartServerRepositoryInsertStream(SmartServerRepositoryInsertStreamLocked):
    """Insert a record stream from a RemoteSink into an unlocked repository.

    This is the same as SmartServerRepositoryInsertStreamLocked, except it
    takes no lock_tokens; i.e. it works with an unlocked (or lock-free, e.g.
    like pack format) repository.

    New in 1.13.
    """

    def do_repository_request(self, repository, resume_tokens):
        """StreamSink.insert_stream for a remote repository."""
        repository.lock_write()
        self.do_insert_stream_request(repository, resume_tokens)