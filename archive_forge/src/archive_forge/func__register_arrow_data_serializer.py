from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _register_arrow_data_serializer(serialization_context):
    """Custom reducer for Arrow data that works around a zero-copy slicing pickling
    bug by using the Arrow IPC format for the underlying serialization.

    Background:
        Arrow has both array-level slicing and buffer-level slicing; both are zero-copy,
        but the former has a serialization bug where the entire buffer is serialized
        instead of just the slice, while the latter's serialization works as expected
        and only serializes the slice of the buffer. I.e., array-level slicing doesn't
        propagate the slice down to the buffer when serializing the array.

        We work around this by registering a custom cloudpickle reducers for Arrow
        Tables that delegates serialization to the Arrow IPC format; thankfully, Arrow's
        IPC serialization has fixed this buffer truncation bug.

    See https://issues.apache.org/jira/browse/ARROW-10739.
    """
    if os.environ.get(RAY_DISABLE_CUSTOM_ARROW_DATA_SERIALIZATION, '0') == '1':
        return
    import pyarrow as pa
    serialization_context._register_cloudpickle_reducer(pa.Table, _arrow_table_reduce)