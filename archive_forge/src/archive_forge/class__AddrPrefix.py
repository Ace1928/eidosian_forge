import abc
import base64
import collections
import copy
import functools
import io
import itertools
import math
import operator
import re
import socket
import struct
import netaddr
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib.packet import afi as addr_family
from os_ken.lib.packet import safi as subaddr_family
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import stream_parser
from os_ken.lib.packet import vxlan
from os_ken.lib.packet import mpls
from os_ken.lib import addrconv
from os_ken.lib import type_desc
from os_ken.lib.type_desc import TypeDisp
from os_ken.lib import ip
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.utils import binary_str
from os_ken.utils import import_module
class _AddrPrefix(StringifyMixin, metaclass=abc.ABCMeta):
    _PACK_STR = '!B'

    def __init__(self, length, addr, prefixes=None, **kwargs):
        assert prefixes != ()
        if isinstance(addr, tuple):
            addr, = addr
        self.length = length
        if prefixes:
            addr = prefixes + (addr,)
        self.addr = addr

    @classmethod
    @abc.abstractmethod
    def _to_bin(cls, addr):
        pass

    @classmethod
    @abc.abstractmethod
    def _from_bin(cls, addr):
        pass

    @classmethod
    def parser(cls, buf):
        length, = struct.unpack_from(cls._PACK_STR, bytes(buf))
        rest = buf[struct.calcsize(cls._PACK_STR):]
        byte_length = (length + 7) // 8
        addr = cls._from_bin(rest[:byte_length])
        rest = rest[byte_length:]
        return (cls(length=length, addr=addr), rest)

    def serialize(self):
        byte_length = (self.length + 7) // 8
        bin_addr = self._to_bin(self.addr)
        if self.length % 8 == 0:
            bin_addr = bin_addr[:byte_length]
        else:
            mask = 65280 >> self.length % 8
            last_byte = struct.Struct('>B').pack(operator.getitem(bin_addr, byte_length - 1) & mask)
            bin_addr = bin_addr[:byte_length - 1] + last_byte
        self.addr = self._from_bin(bin_addr)
        buf = bytearray()
        msg_pack_into(self._PACK_STR, buf, 0, self.length)
        return buf + bytes(bin_addr)