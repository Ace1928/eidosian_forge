import operator
import os
from io import BytesIO
from ..lazy_import import lazy_import
import patiencediff
import gzip
from breezy import (
from breezy.bzr import (
from breezy.bzr import pack_repo
from breezy.i18n import gettext
from .. import annotate, errors, osutils
from .. import transport as _mod_transport
from ..bzr.versionedfile import (AbsentContentFactory, ConstantMapper,
from ..errors import InternalBzrError, InvalidRevisionId, RevisionNotPresent
from ..osutils import contains_whitespace, sha_string, sha_strings, split_lines
from ..transport import NoSuchFile
from . import index as _mod_index
class _KnitKeyAccess:
    """Access to records in .knit files."""

    def __init__(self, transport, mapper):
        """Create a _KnitKeyAccess with transport and mapper.

        :param transport: The transport the access object is rooted at.
        :param mapper: The mapper used to map keys to .knit files.
        """
        self._transport = transport
        self._mapper = mapper

    def add_raw_record(self, key, size, raw_data):
        """Add raw knit bytes to a storage area.

        The data is spooled to the container writer in one bytes-record per
        raw data item.

        :param key: The key of the raw data segment
        :param size: The size of the raw data segment
        :param raw_data: A chunked bytestring containing the data.
        :return: opaque index memo to retrieve the record later.
            For _KnitKeyAccess the memo is (key, pos, length), where the key is
            the record key.
        """
        path = self._mapper.map(key)
        try:
            base = self._transport.append_bytes(path + '.knit', b''.join(raw_data))
        except _mod_transport.NoSuchFile:
            self._transport.mkdir(osutils.dirname(path))
            base = self._transport.append_bytes(path + '.knit', b''.join(raw_data))
        return (key, base, size)

    def add_raw_records(self, key_sizes, raw_data):
        """Add raw knit bytes to a storage area.

        The data is spooled to the container writer in one bytes-record per
        raw data item.

        :param sizes: An iterable of tuples containing the key and size of each
            raw data segment.
        :param raw_data: A chunked bytestring containing the data.
        :return: A list of memos to retrieve the record later. Each memo is an
            opaque index memo. For _KnitKeyAccess the memo is (key, pos,
            length), where the key is the record key.
        """
        raw_data = b''.join(raw_data)
        if not isinstance(raw_data, bytes):
            raise AssertionError('data must be plain bytes was %s' % type(raw_data))
        result = []
        offset = 0
        for key, size in key_sizes:
            record_bytes = [raw_data[offset:offset + size]]
            result.append(self.add_raw_record(key, size, record_bytes))
            offset += size
        return result

    def flush(self):
        """Flush pending writes on this access object.

        For .knit files this is a no-op.
        """
        pass

    def get_raw_records(self, memos_for_retrieval):
        """Get the raw bytes for a records.

        :param memos_for_retrieval: An iterable containing the access memo for
            retrieving the bytes.
        :return: An iterator over the bytes of the records.
        """
        request_lists = []
        current_prefix = None
        for key, offset, length in memos_for_retrieval:
            if current_prefix == key[:-1]:
                current_list.append((offset, length))
            else:
                if current_prefix is not None:
                    request_lists.append((current_prefix, current_list))
                current_prefix = key[:-1]
                current_list = [(offset, length)]
        if current_prefix is not None:
            request_lists.append((current_prefix, current_list))
        for prefix, read_vector in request_lists:
            path = self._mapper.map(prefix) + '.knit'
            for pos, data in self._transport.readv(path, read_vector):
                yield data