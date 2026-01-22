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
class AESCompressor(ISevenZipCompressor):
    """AES Compression(Encryption) class.
    It accept pre-processing filter which may be a LZMA compression."""
    AES_CBC_BLOCKSIZE = 16

    def __init__(self, password: str, blocksize: Optional[int]=None) -> None:
        self.cycles = 19
        self.iv = get_random_bytes(16)
        self.salt = b''
        self.method = CompressionMethod.CRYPT_AES256_SHA256
        key = calculate_key(password.encode('utf-16LE'), self.cycles, self.salt, 'sha256')
        self.iv += bytes(self.AES_CBC_BLOCKSIZE - len(self.iv))
        self.cipher = AES.new(key, AES.MODE_CBC, self.iv)
        self.flushed = False
        if blocksize:
            self.buf = Buffer(size=blocksize + self.AES_CBC_BLOCKSIZE * 2)
        else:
            self.buf = Buffer(size=get_default_blocksize() + self.AES_CBC_BLOCKSIZE * 2)

    def encode_filter_properties(self):
        saltsize = len(self.salt)
        ivsize = len(self.iv)
        ivfirst = 1
        saltfirst = 1 if len(self.salt) > 0 else 0
        firstbyte = (self.cycles + (ivfirst << 6) + (saltfirst << 7)).to_bytes(1, 'little')
        secondbyte = ((ivsize - 1 & 15) + (saltsize - saltfirst << 4 & 240)).to_bytes(1, 'little')
        properties = firstbyte + secondbyte + self.salt + self.iv
        return properties

    def compress(self, data):
        """Compression + AES encryption with 16byte alignment."""
        currentlen = len(self.buf) + len(data)
        if currentlen >= 16 and currentlen & 15 == 0:
            self.buf.add(data)
            res = self.cipher.encrypt(self.buf.view)
            self.buf.reset()
        elif currentlen > 16:
            nextpos = currentlen & ~15
            buflen = len(self.buf)
            self.buf.add(data[:nextpos - buflen])
            res = self.cipher.encrypt(self.buf.view)
            self.buf.set(data[nextpos - buflen:])
        else:
            self.buf.add(data)
            res = b''
        return res

    def flush(self):
        if len(self.buf) > 0:
            padlen = -len(self.buf) & 15
            self.buf.add(bytes(padlen))
            res = self.cipher.encrypt(self.buf.view)
            self.buf.reset()
        else:
            res = b''
        return res