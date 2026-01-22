import logging
import os
import struct
import zlib
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
def _read_header(self):
    header = self._fp.read(LEVELDBLOG_HEADER_LEN)
    assert len(header) == LEVELDBLOG_HEADER_LEN, 'header is {} bytes instead of the expected {}'.format(len(header), LEVELDBLOG_HEADER_LEN)
    ident, magic, version = struct.unpack('<4sHB', header)
    if ident != strtobytes(LEVELDBLOG_HEADER_IDENT):
        raise Exception('Invalid header')
    if magic != LEVELDBLOG_HEADER_MAGIC:
        raise Exception('Invalid header')
    if version != LEVELDBLOG_HEADER_VERSION:
        raise Exception('Invalid header')
    self._index += len(header)