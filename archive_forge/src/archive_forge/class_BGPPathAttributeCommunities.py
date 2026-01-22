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
@_PathAttribute.register_type(BGP_ATTR_TYPE_COMMUNITIES)
class BGPPathAttributeCommunities(_PathAttribute):
    _VALUE_PACK_STR = '!I'
    _ATTR_FLAGS = BGP_ATTR_FLAG_OPTIONAL | BGP_ATTR_FLAG_TRANSITIVE
    NO_EXPORT = int('0xFFFFFF01', 16)
    NO_ADVERTISE = int('0xFFFFFF02', 16)
    NO_EXPORT_SUBCONFED = int('0xFFFFFF03', 16)
    WELL_KNOW_COMMUNITIES = (NO_EXPORT, NO_ADVERTISE, NO_EXPORT_SUBCONFED)

    def __init__(self, communities, flags=0, type_=None, length=None):
        super(BGPPathAttributeCommunities, self).__init__(flags=flags, type_=type_, length=length)
        self.communities = communities

    @classmethod
    def parse_value(cls, buf):
        rest = buf
        communities = []
        elem_size = struct.calcsize(cls._VALUE_PACK_STR)
        while len(rest) >= elem_size:
            comm, = struct.unpack_from(cls._VALUE_PACK_STR, bytes(rest))
            communities.append(comm)
            rest = rest[elem_size:]
        return {'communities': communities}

    def serialize_value(self):
        buf = bytearray()
        for comm in self.communities:
            bincomm = bytearray()
            msg_pack_into(self._VALUE_PACK_STR, bincomm, 0, comm)
            buf += bincomm
        return buf

    @staticmethod
    def is_no_export(comm_attr):
        """Returns True if given value matches well-known community NO_EXPORT
         attribute value.
         """
        return comm_attr == BGPPathAttributeCommunities.NO_EXPORT

    @staticmethod
    def is_no_advertise(comm_attr):
        """Returns True if given value matches well-known community
        NO_ADVERTISE attribute value.
        """
        return comm_attr == BGPPathAttributeCommunities.NO_ADVERTISE

    @staticmethod
    def is_no_export_subconfed(comm_attr):
        """Returns True if given value matches well-known community
         NO_EXPORT_SUBCONFED attribute value.
         """
        return comm_attr == BGPPathAttributeCommunities.NO_EXPORT_SUBCONFED

    def has_comm_attr(self, attr):
        """Returns True if given community attribute is present."""
        for comm_attr in self.communities:
            if comm_attr == attr:
                return True
        return False