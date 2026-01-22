from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _restore_table_from_ipc(buf: bytes) -> 'pyarrow.Table':
    """Restore an Arrow Table serialized to Arrow IPC format."""
    from pyarrow.ipc import RecordBatchStreamReader
    with RecordBatchStreamReader(buf) as reader:
        return reader.read_all()