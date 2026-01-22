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
class _BGPPathAttributeAsPathCommon(_PathAttribute):
    _AS_SET = 1
    _AS_SEQUENCE = 2
    _SEG_HDR_PACK_STR = '!BB'
    _AS_PACK_STR = None
    _ATTR_FLAGS = BGP_ATTR_FLAG_TRANSITIVE

    def __init__(self, value, as_pack_str=None, flags=0, type_=None, length=None):
        super(_BGPPathAttributeAsPathCommon, self).__init__(value=value, flags=flags, type_=type_, length=length)
        if as_pack_str:
            self._AS_PACK_STR = as_pack_str

    @property
    def path_seg_list(self):
        return copy.deepcopy(self.value)

    def get_as_path_len(self):
        count = 0
        for seg in self.value:
            if isinstance(seg, list):
                count += len(seg)
            else:
                count += 1
        return count

    def has_local_as(self, local_as, max_count=0):
        """Check if *local_as* is already present on path list."""
        _count = 0
        for as_path_seg in self.value:
            _count += list(as_path_seg).count(local_as)
        return _count > max_count

    def has_matching_leftmost(self, remote_as):
        """Check if leftmost AS matches *remote_as*."""
        if not self.value or not remote_as:
            return False
        leftmost_seg = self.path_seg_list[0]
        if leftmost_seg and leftmost_seg[0] == remote_as:
            return True
        return False

    @classmethod
    def _is_valid_16bit_as_path(cls, buf):
        two_byte_as_size = struct.calcsize('!H')
        while buf:
            type_, num_as = struct.unpack_from(cls._SEG_HDR_PACK_STR, bytes(buf))
            if type_ is not cls._AS_SET and type_ is not cls._AS_SEQUENCE:
                return False
            buf = buf[struct.calcsize(cls._SEG_HDR_PACK_STR):]
            if len(buf) < num_as * two_byte_as_size:
                return False
            buf = buf[num_as * two_byte_as_size:]
        return True

    @classmethod
    def parse_value(cls, buf):
        result = []
        if cls._is_valid_16bit_as_path(buf):
            as_pack_str = '!H'
        else:
            as_pack_str = '!I'
        while buf:
            type_, num_as = struct.unpack_from(cls._SEG_HDR_PACK_STR, bytes(buf))
            buf = buf[struct.calcsize(cls._SEG_HDR_PACK_STR):]
            l = []
            for _ in range(0, num_as):
                as_number, = struct.unpack_from(as_pack_str, bytes(buf))
                buf = buf[struct.calcsize(as_pack_str):]
                l.append(as_number)
            if type_ == cls._AS_SET:
                result.append(set(l))
            elif type_ == cls._AS_SEQUENCE:
                result.append(l)
            else:
                raise struct.error('Unsupported segment type: %s' % type_)
        return {'value': result, 'as_pack_str': as_pack_str}

    def serialize_value(self):
        buf = bytearray()
        offset = 0
        for e in self.value:
            if isinstance(e, set):
                type_ = self._AS_SET
            elif isinstance(e, list):
                type_ = self._AS_SEQUENCE
            else:
                raise struct.error('Element of %s.value must be of type set or list' % self.__class__.__name__)
            l = list(e)
            num_as = len(l)
            if num_as == 0:
                continue
            msg_pack_into(self._SEG_HDR_PACK_STR, buf, offset, type_, num_as)
            offset += struct.calcsize(self._SEG_HDR_PACK_STR)
            for i in l:
                msg_pack_into(self._AS_PACK_STR, buf, offset, i)
                offset += struct.calcsize(self._AS_PACK_STR)
        return buf