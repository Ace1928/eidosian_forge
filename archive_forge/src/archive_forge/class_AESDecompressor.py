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
class AESDecompressor(ISevenZipDecompressor):
    """Decrypt data"""

    def __init__(self, aes_properties: bytes, password: str, blocksize: Optional[int]=None) -> None:
        firstbyte = aes_properties[0]
        numcyclespower = firstbyte & 63
        if firstbyte & 192 != 0:
            saltsize = firstbyte >> 7 & 1
            ivsize = firstbyte >> 6 & 1
            secondbyte = aes_properties[1]
            saltsize += secondbyte >> 4
            ivsize += secondbyte & 15
            assert len(aes_properties) == 2 + saltsize + ivsize
            salt = aes_properties[2:2 + saltsize]
            iv = aes_properties[2 + saltsize:2 + saltsize + ivsize]
            assert len(salt) == saltsize
            assert len(iv) == ivsize
            assert numcyclespower <= 24
            if ivsize < 16:
                iv += bytes('\x00' * (16 - ivsize), 'ascii')
            key = calculate_key(password.encode('utf-16LE'), numcyclespower, salt, 'sha256')
            self.cipher = AES.new(key, AES.MODE_CBC, iv)
            if blocksize:
                self.buf = Buffer(size=blocksize + 16)
            else:
                self.buf = Buffer(size=get_default_blocksize() + 16)
        else:
            raise UnsupportedCompressionMethodError(firstbyte, 'Wrong 7zAES properties')

    def decompress(self, data: Union[bytes, bytearray, memoryview], max_length: int=-1) -> bytes:
        currentlen = len(self.buf) + len(data)
        if len(data) > 0 and currentlen & 15 == 0:
            self.buf.add(data)
            temp = self.cipher.decrypt(self.buf.view)
            self.buf.reset()
            return temp
        elif len(data) > 0:
            nextpos = currentlen & ~15
            buflen = len(self.buf)
            temp2 = data[nextpos - buflen:]
            self.buf.add(data[:nextpos - buflen])
            temp = self.cipher.decrypt(self.buf.view)
            self.buf.set(temp2)
            return temp
        elif len(self.buf) == 0:
            return b''
        else:
            padlen = -len(self.buf) & 15
            self.buf.add(bytes(padlen))
            temp3 = self.cipher.decrypt(self.buf.view)
            self.buf.reset()
            return temp3