import os
import pyarrow as pa
from pyarrow.lib import (IpcReadOptions, IpcWriteOptions, ReadStats, WriteStats,  # noqa
import pyarrow.lib as lib
class RecordBatchFileWriter(lib._RecordBatchFileWriter):
    __doc__ = 'Writer to create the Arrow binary file format\n\n{}'.format(_ipc_writer_class_doc)

    def __init__(self, sink, schema, *, use_legacy_format=None, options=None):
        options = _get_legacy_format_default(use_legacy_format, options)
        self._open(sink, schema, options=options)