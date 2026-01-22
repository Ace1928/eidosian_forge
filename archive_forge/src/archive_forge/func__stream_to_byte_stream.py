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
def _stream_to_byte_stream(stream, src_format):
    """Convert a record stream to a self delimited byte stream."""
    pack_writer = pack.ContainerSerialiser()
    yield pack_writer.begin()
    yield pack_writer.bytes_record(src_format.network_name(), b'')
    for substream_type, substream in stream:
        for record in substream:
            if record.storage_kind in ('chunked', 'fulltext'):
                serialised = record_to_fulltext_bytes(record)
            elif record.storage_kind == 'absent':
                raise ValueError('Absent factory for {}'.format(record.key))
            else:
                serialised = record.get_bytes_as(record.storage_kind)
            if serialised:
                yield pack_writer.bytes_record(serialised, [(substream_type.encode('ascii'),)])
    yield pack_writer.end()