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
class _FlowSpecComponentBase(StringifyMixin, TypeDisp):
    """
    Base class for Flow Specification NLRI component
    """
    COMPONENT_NAME = None
    _BASE_STR = '!B'
    _BASE_STR_SIZE = struct.calcsize(_BASE_STR)
    _NAMES = {}

    def __init__(self, type_=None):
        if type_ is None:
            type_, _ = self._rev_lookup_type(self.__class__)
        self.type = type_

    @classmethod
    def register_type(cls, type_, afi):
        cls._TYPES = cls._TYPES.copy()
        cls._NAMES = cls._NAMES.copy()

        def _register_type(subcls):
            cls._TYPES[type_, afi] = subcls
            cls._NAMES[subcls.COMPONENT_NAME, afi] = subcls
            cls._REV_TYPES = None
            return subcls
        return _register_type

    @classmethod
    def lookup_type_name(cls, type_name, afi):
        return cls._NAMES[type_name, afi]

    @classmethod
    def _lookup_type(cls, type_, afi):
        try:
            return cls._TYPES[type_, afi]
        except KeyError:
            return cls._UNKNOWN_TYPE

    @classmethod
    def parse_header(cls, rest, afi):
        type_, = struct.unpack_from(cls._BASE_STR, bytes(rest))
        rest = rest[cls._BASE_STR_SIZE:]
        return (cls._lookup_type(type_, afi), rest)

    def serialize_header(self):
        return struct.pack(self._BASE_STR, self.type)