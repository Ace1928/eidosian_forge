import bz2
import lzma
import struct
import sys
import zlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import bcj
import inflate64
import pyppmd
import pyzstd
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from py7zr.exceptions import PasswordRequired, UnsupportedCompressionMethodError
from py7zr.helpers import Buffer, calculate_crc32, calculate_key
from py7zr.properties import (
class Deflate64Compressor(ISevenZipCompressor):

    def __init__(self):
        self.flushed = False
        if hasattr(sys, 'pypy_version_info'):
            self._enabled = False
        else:
            self._compressor = inflate64.Deflater()
            self._enabled = True

    def compress(self, data: Union[bytes, bytearray, memoryview], max_length: int=-1) -> bytes:
        if not self._enabled:
            raise UnsupportedCompressionMethodError(None, 'deflate64 is disabled on pypy.')
        return self._compressor.deflate(data)

    def flush(self) -> bytes:
        if not self._enabled:
            raise UnsupportedCompressionMethodError(None, 'deflate64 is disabled on pypy.')
        if self.flushed:
            return b''
        self.flushed = True
        return self._compressor.flush()