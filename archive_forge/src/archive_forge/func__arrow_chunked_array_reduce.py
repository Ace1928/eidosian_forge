from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _arrow_chunked_array_reduce(ca: 'pyarrow.ChunkedArray') -> Tuple[List['PicklableArrayPayload'], 'pyarrow.DataType']:
    """Custom reducer for Arrow ChunkedArrays that works around a zero-copy slice
    pickling bug. This reducer does not return a reconstruction function, since it's
    expected to be reconstructed by the Arrow Table reconstructor.
    """
    chunk_payloads = []
    for chunk in ca.chunks:
        chunk_payload = PicklableArrayPayload.from_array(chunk)
        chunk_payloads.append(chunk_payload)
    return (chunk_payloads, ca.type)