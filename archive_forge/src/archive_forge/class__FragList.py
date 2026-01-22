from __future__ import annotations
import binascii
import enum
import os
import re
import typing
import warnings
from base64 import encodebytes as _base64_encode
from dataclasses import dataclass
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.ciphers import (
from cryptography.hazmat.primitives.serialization import (
class _FragList:
    """Build recursive structure without data copy."""
    flist: typing.List[bytes]

    def __init__(self, init: typing.Optional[typing.List[bytes]]=None) -> None:
        self.flist = []
        if init:
            self.flist.extend(init)

    def put_raw(self, val: bytes) -> None:
        """Add plain bytes"""
        self.flist.append(val)

    def put_u32(self, val: int) -> None:
        """Big-endian uint32"""
        self.flist.append(val.to_bytes(length=4, byteorder='big'))

    def put_u64(self, val: int) -> None:
        """Big-endian uint64"""
        self.flist.append(val.to_bytes(length=8, byteorder='big'))

    def put_sshstr(self, val: typing.Union[bytes, _FragList]) -> None:
        """Bytes prefixed with u32 length"""
        if isinstance(val, (bytes, memoryview, bytearray)):
            self.put_u32(len(val))
            self.flist.append(val)
        else:
            self.put_u32(val.size())
            self.flist.extend(val.flist)

    def put_mpint(self, val: int) -> None:
        """Big-endian bigint prefixed with u32 length"""
        self.put_sshstr(_to_mpint(val))

    def size(self) -> int:
        """Current number of bytes"""
        return sum(map(len, self.flist))

    def render(self, dstbuf: memoryview, pos: int=0) -> int:
        """Write into bytearray"""
        for frag in self.flist:
            flen = len(frag)
            start, pos = (pos, pos + flen)
            dstbuf[start:pos] = frag
        return pos

    def tobytes(self) -> bytes:
        """Return as bytes"""
        buf = memoryview(bytearray(self.size()))
        self.render(buf)
        return buf.tobytes()