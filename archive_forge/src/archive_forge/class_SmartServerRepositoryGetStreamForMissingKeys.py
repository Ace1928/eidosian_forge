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
class SmartServerRepositoryGetStreamForMissingKeys(SmartServerRepositoryRequest):

    def do_repository_request(self, repository, to_network_name):
        """Get a stream for missing keys.

        :param repository: The repository to stream from.
        :param to_network_name: The network name of the format of the target
            repository.
        """
        try:
            self._to_format = network_format_registry.get(to_network_name)
        except KeyError:
            return FailedSmartServerResponse((b'UnknownFormat', b'repository', to_network_name))
        return None

    def do_body(self, body_bytes):
        repository = self._repository
        repository.lock_read()
        try:
            source = repository._get_source(self._to_format)
            keys = []
            for entry in body_bytes.split(b'\n'):
                kind, revid = entry.split(b'\t')
                keys.append((kind.decode('utf-8'), revid))
            stream = source.get_stream_for_missing_keys(keys)
        except Exception:
            try:
                repository.unlock()
            finally:
                raise
        return SuccessfulSmartServerResponse((b'ok',), body_stream=self.body_stream(stream, repository))

    def body_stream(self, stream, repository):
        byte_stream = _stream_to_byte_stream(stream, repository._format)
        try:
            yield from byte_stream
        except errors.RevisionNotPresent as e:
            repository.unlock()
            yield FailedSmartServerResponse((b'NoSuchRevision', e.revision_id))
        else:
            repository.unlock()