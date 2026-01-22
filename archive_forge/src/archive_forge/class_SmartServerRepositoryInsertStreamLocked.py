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
class SmartServerRepositoryInsertStreamLocked(SmartServerRepositoryRequest):
    """Insert a record stream from a RemoteSink into a repository.

    This gets bytes pushed to it by the network infrastructure and turns that
    into a bytes iterator using a thread. That is then processed by
    _byte_stream_to_stream.

    New in 1.14.
    """

    def do_repository_request(self, repository, resume_tokens, lock_token):
        """StreamSink.insert_stream for a remote repository."""
        repository.lock_write(token=lock_token)
        self.do_insert_stream_request(repository, resume_tokens)

    def do_insert_stream_request(self, repository, resume_tokens):
        tokens = [token.decode('utf-8') for token in resume_tokens.split(b' ') if token]
        self.tokens = tokens
        self.repository = repository
        self.queue = queue.Queue()
        self.insert_thread = threading.Thread(target=self._inserter_thread)
        self.insert_thread.start()

    def do_chunk(self, body_stream_chunk):
        self.queue.put(body_stream_chunk)

    def _inserter_thread(self):
        try:
            src_format, stream = _byte_stream_to_stream(self.blocking_byte_stream())
            self.insert_result = self.repository._get_sink().insert_stream(stream, src_format, self.tokens)
            self.insert_ok = True
        except:
            self.insert_exception = sys.exc_info()
            self.insert_ok = False

    def blocking_byte_stream(self):
        while True:
            bytes = self.queue.get()
            if bytes is StopIteration:
                return
            else:
                yield bytes

    def do_end(self):
        self.queue.put(StopIteration)
        if self.insert_thread is not None:
            self.insert_thread.join()
        if not self.insert_ok:
            exc_type, exc_val, exc_tb = self.insert_exception
            try:
                raise exc_val
            finally:
                del self.insert_exception
        write_group_tokens, missing_keys = self.insert_result
        if write_group_tokens or missing_keys:
            missing_keys = sorted([(entry[0].encode('utf-8'),) + entry[1:] for entry in missing_keys])
            bytes = bencode.bencode(([token.encode('utf-8') for token in write_group_tokens], missing_keys))
            self.repository.unlock()
            return SuccessfulSmartServerResponse((b'missing-basis', bytes))
        else:
            self.repository.unlock()
            return SuccessfulSmartServerResponse((b'ok',))