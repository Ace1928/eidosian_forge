import os
import pyarrow as pa
from pyarrow.lib import (IpcReadOptions, IpcWriteOptions, ReadStats, WriteStats,  # noqa
import pyarrow.lib as lib
class RecordBatchFileReader(lib._RecordBatchFileReader):
    """
    Class for reading Arrow record batch data from the Arrow binary file format

    Parameters
    ----------
    source : bytes/buffer-like, pyarrow.NativeFile, or file-like Python object
        Either an in-memory buffer, or a readable file object.
        If you want to use memory map use MemoryMappedFile as source.
    footer_offset : int, default None
        If the file is embedded in some larger file, this is the byte offset to
        the very end of the file data
    options : pyarrow.ipc.IpcReadOptions
        Options for IPC serialization.
        If None, default values will be used.
    memory_pool : MemoryPool, default None
        If None, default memory pool is used.
    """

    def __init__(self, source, footer_offset=None, *, options=None, memory_pool=None):
        options = _ensure_default_ipc_read_options(options)
        self._open(source, footer_offset=footer_offset, options=options, memory_pool=memory_pool)