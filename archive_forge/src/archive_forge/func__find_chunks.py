import os
import json
import struct
import logging
import numpy as np
from ..core import Format
from ..v2 import imread
def _find_chunks(self):
    """
            Gets start position and size of data chunks in file.
            """
    chunk_header = b'\x89LFC\r\n\x1a\n\x00\x00\x00\x00'
    for i in range(0, DATA_CHUNKS_F01):
        data_pos, size, sha1 = self._get_chunk(chunk_header)
        self._chunks[sha1] = (data_pos, size)