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
@property
def unpacksizes(self) -> List[int]:
    result: List[int] = []
    shift = 0
    prev = False
    for i, r in enumerate(self.methods_map):
        shift += 1 if r and prev else 0
        prev = r
        result.insert(0, self._unpacksizes[i - shift])
    return result