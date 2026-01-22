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
class BGPMessage(packet_base.PacketBase, TypeDisp):
    """Base class for BGP-4 messages.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    ========================== ===============================================
    Attribute                  Description
    ========================== ===============================================
    marker                     Marker field.  Ignored when encoding.
    len                        Length field.  Ignored when encoding.
    type                       Type field.  one of ``BGP_MSG_*`` constants.
    ========================== ===============================================
    """
    _HDR_PACK_STR = '!16sHB'
    _HDR_LEN = struct.calcsize(_HDR_PACK_STR)
    _class_prefixes = ['BGP']

    def __init__(self, marker=None, len_=None, type_=None):
        super(BGPMessage, self).__init__()
        if marker is None:
            self._marker = _MARKER
        else:
            self._marker = marker
        self.len = len_
        if type_ is None:
            type_ = self._rev_lookup_type(self.__class__)
        self.type = type_

    @classmethod
    def parser(cls, buf):
        if len(buf) < cls._HDR_LEN:
            raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), cls._HDR_LEN))
        marker, len_, type_ = struct.unpack_from(cls._HDR_PACK_STR, bytes(buf))
        msglen = len_
        if len(buf) < msglen:
            raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), msglen))
        binmsg = buf[cls._HDR_LEN:msglen]
        rest = buf[msglen:]
        subcls = cls._lookup_type(type_)
        kwargs = subcls.parser(binmsg)
        return (subcls(marker=marker, len_=len_, type_=type_, **kwargs), cls, rest)

    def serialize(self, payload=None, prev=None):
        self._marker = _MARKER
        tail = self.serialize_tail()
        self.len = self._HDR_LEN + len(tail)
        hdr = bytearray(struct.pack(self._HDR_PACK_STR, self._marker, self.len, self.type))
        return hdr + tail

    def __len__(self):
        buf = self.serialize()
        return len(buf)