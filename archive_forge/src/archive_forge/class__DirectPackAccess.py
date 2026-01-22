import re
import sys
from typing import Type
from ..lazy_import import lazy_import
import contextlib
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.index import (
from .. import errors, lockable_files, lockdir
from .. import transport as _mod_transport
from ..bzr import btree_index, index
from ..decorators import only_raises
from ..lock import LogicalLockResult
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..trace import mutter, note, warning
from .repository import MetaDirRepository, RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (MetaDirVersionedFileRepository,
class _DirectPackAccess:
    """Access to data in one or more packs with less translation."""

    def __init__(self, index_to_packs, reload_func=None, flush_func=None):
        """Create a _DirectPackAccess object.

        :param index_to_packs: A dict mapping index objects to the transport
            and file names for obtaining data.
        :param reload_func: A function to call if we determine that the pack
            files have moved and we need to reload our caches. See
            breezy.repo_fmt.pack_repo.AggregateIndex for more details.
        """
        self._container_writer = None
        self._write_index = None
        self._indices = index_to_packs
        self._reload_func = reload_func
        self._flush_func = flush_func

    def add_raw_record(self, key, size, raw_data):
        """Add raw knit bytes to a storage area.

        The data is spooled to the container writer in one bytes-record per
        raw data item.

        :param key: key of the data segment
        :param size: length of the data segment
        :param raw_data: A bytestring containing the data.
        :return: An opaque index memo For _DirectPackAccess the memo is
            (index, pos, length), where the index field is the write_index
            object supplied to the PackAccess object.
        """
        p_offset, p_length = self._container_writer.add_bytes_record(raw_data, size, [])
        return (self._write_index, p_offset, p_length)

    def add_raw_records(self, key_sizes, raw_data):
        """Add raw knit bytes to a storage area.

        The data is spooled to the container writer in one bytes-record per
        raw data item.

        :param sizes: An iterable of tuples containing the key and size of each
            raw data segment.
        :param raw_data: A bytestring containing the data.
        :return: A list of memos to retrieve the record later. Each memo is an
            opaque index memo. For _DirectPackAccess the memo is (index, pos,
            length), where the index field is the write_index object supplied
            to the PackAccess object.
        """
        raw_data = b''.join(raw_data)
        if not isinstance(raw_data, bytes):
            raise AssertionError('data must be plain bytes was %s' % type(raw_data))
        result = []
        offset = 0
        for key, size in key_sizes:
            result.append(self.add_raw_record(key, size, [raw_data[offset:offset + size]]))
            offset += size
        return result

    def flush(self):
        """Flush pending writes on this access object.

        This will flush any buffered writes to a NewPack.
        """
        if self._flush_func is not None:
            self._flush_func()

    def get_raw_records(self, memos_for_retrieval):
        """Get the raw bytes for a records.

        :param memos_for_retrieval: An iterable containing the (index, pos,
            length) memo for retrieving the bytes. The Pack access method
            looks up the pack to use for a given record in its index_to_pack
            map.
        :return: An iterator over the bytes of the records.
        """
        request_lists = []
        current_index = None
        for index, offset, length in memos_for_retrieval:
            if current_index == index:
                current_list.append((offset, length))
            else:
                if current_index is not None:
                    request_lists.append((current_index, current_list))
                current_index = index
                current_list = [(offset, length)]
        if current_index is not None:
            request_lists.append((current_index, current_list))
        for index, offsets in request_lists:
            try:
                transport, path = self._indices[index]
            except KeyError:
                if self._reload_func is None:
                    raise
                raise RetryWithNewPacks(index, reload_occurred=True, exc_info=sys.exc_info())
            try:
                reader = pack.make_readv_reader(transport, path, offsets)
                for names, read_func in reader.iter_records():
                    yield read_func(None)
            except _mod_transport.NoSuchFile:
                if self._reload_func is None:
                    raise
                raise RetryWithNewPacks(transport.abspath(path), reload_occurred=False, exc_info=sys.exc_info())

    def set_writer(self, writer, index, transport_packname):
        """Set a writer to use for adding data."""
        if index is not None:
            self._indices[index] = transport_packname
        self._container_writer = writer
        self._write_index = index

    def reload_or_raise(self, retry_exc):
        """Try calling the reload function, or re-raise the original exception.

        This should be called after _DirectPackAccess raises a
        RetryWithNewPacks exception. This function will handle the common logic
        of determining when the error is fatal versus being temporary.
        It will also make sure that the original exception is raised, rather
        than the RetryWithNewPacks exception.

        If this function returns, then the calling function should retry
        whatever operation was being performed. Otherwise an exception will
        be raised.

        :param retry_exc: A RetryWithNewPacks exception.
        """
        is_error = False
        if self._reload_func is None:
            is_error = True
        elif not self._reload_func():
            if not retry_exc.reload_occurred:
                is_error = True
        if is_error:
            raise retry_exc.exc_info[1]