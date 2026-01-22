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
class _ByteStreamDecoder:
    """Helper for _byte_stream_to_stream.

    The expected usage of this class is via the function _byte_stream_to_stream
    which creates a _ByteStreamDecoder, pops off the stream format and then
    yields the output of record_stream(), the main entry point to
    _ByteStreamDecoder.

    Broadly this class has to unwrap two layers of iterators:
    (type, substream)
    (substream details)

    This is complicated by wishing to return type, iterator_for_type, but
    getting the data for iterator_for_type when we find out type: we can't
    simply pass a generator down to the NetworkRecordStream parser, instead
    we have a little local state to seed each NetworkRecordStream instance,
    and gather the type that we'll be yielding.

    :ivar byte_stream: The byte stream being decoded.
    :ivar stream_decoder: A pack parser used to decode the bytestream
    :ivar current_type: The current type, used to join adjacent records of the
        same type into a single stream.
    :ivar first_bytes: The first bytes to give the next NetworkRecordStream.
    """

    def __init__(self, byte_stream, record_counter):
        """Create a _ByteStreamDecoder."""
        self.stream_decoder = pack.ContainerPushParser()
        self.current_type = None
        self.first_bytes = None
        self.byte_stream = byte_stream
        self._record_counter = record_counter
        self.key_count = 0

    def iter_stream_decoder(self):
        """Iterate the contents of the pack from stream_decoder."""
        yield from self.stream_decoder.read_pending_records()
        for bytes in self.byte_stream:
            self.stream_decoder.accept_bytes(bytes)
            yield from self.stream_decoder.read_pending_records()

    def iter_substream_bytes(self):
        if self.first_bytes is not None:
            yield self.first_bytes
            self.first_bytes = None
        for record in self.iter_pack_records:
            record_names, record_bytes = record
            record_name, = record_names
            substream_type = record_name[0]
            if substream_type != self.current_type:
                self.current_type = substream_type
                self.first_bytes = record_bytes
                return
            yield record_bytes

    def record_stream(self):
        """Yield substream_type, substream from the byte stream."""

        def wrap_and_count(pb, rc, substream):
            """Yield records from stream while showing progress."""
            counter = 0
            if rc:
                if self.current_type != 'revisions' and self.key_count != 0:
                    if not rc.is_initialized():
                        rc.setup(self.key_count, self.key_count)
            for record in substream.read():
                if rc:
                    if rc.is_initialized() and counter == rc.STEP:
                        rc.increment(counter)
                        pb.update('Estimate', rc.current, rc.max)
                        counter = 0
                    if self.current_type == 'revisions':
                        self.key_count += 1
                        if counter == rc.STEP:
                            pb.update('Estimating..', self.key_count)
                            counter = 0
                counter += 1
                yield record
        self.seed_state()
        with ui.ui_factory.nested_progress_bar() as pb:
            rc = self._record_counter
            try:
                while self.first_bytes is not None:
                    substream = NetworkRecordStream(self.iter_substream_bytes())
                    yield (self.current_type.decode('ascii'), wrap_and_count(pb, rc, substream))
            finally:
                if rc:
                    pb.update('Done', rc.max, rc.max)

    def seed_state(self):
        """Prepare the _ByteStreamDecoder to decode from the pack stream."""
        self.iter_pack_records = self.iter_stream_decoder()
        list(self.iter_substream_bytes())