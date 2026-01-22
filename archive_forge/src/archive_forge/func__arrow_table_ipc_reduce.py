from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _arrow_table_ipc_reduce(table: 'pyarrow.Table'):
    """Custom reducer for Arrow Table that works around a zero-copy slicing pickling
    bug by using the Arrow IPC format for the underlying serialization.

    This is currently used as a fallback for unsupported types (or unknown bugs) for
    the manual buffer truncation workaround, e.g. for dense unions.
    """
    from pyarrow.ipc import RecordBatchStreamWriter
    from pyarrow.lib import BufferOutputStream
    output_stream = BufferOutputStream()
    with RecordBatchStreamWriter(output_stream, schema=table.schema) as wr:
        wr.write_table(table)
    return (_restore_table_from_ipc, (output_stream.getvalue(),))