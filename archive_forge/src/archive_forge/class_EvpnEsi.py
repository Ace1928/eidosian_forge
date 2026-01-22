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
class EvpnEsi(StringifyMixin, TypeDisp, _Value):
    """
    Ethernet Segment Identifier

    The supported ESI Types:

     - ``EvpnEsi.ARBITRARY`` indicates EvpnArbitraryEsi.

     - ``EvpnEsi.LACP`` indicates EvpnLACPEsi.

     - ``EvpnEsi.L2_BRIDGE`` indicates EvpnL2BridgeEsi.

     - ``EvpnEsi.MAC_BASED`` indicates EvpnMacBasedEsi.

     - ``EvpnEsi.ROUTER_ID`` indicates EvpnRouterIDEsi.

     - ``EvpnEsi.AS_BASED`` indicates EvpnASBasedEsi.
    """
    _PACK_STR = '!B'
    _ESI_LEN = 10
    ARBITRARY = 0
    LACP = 1
    L2_BRIDGE = 2
    MAC_BASED = 3
    ROUTER_ID = 4
    AS_BASED = 5
    MAX = 255
    _TYPE_NAME = None

    def __init__(self, type_=None):
        if type_ is None:
            type_ = self._rev_lookup_type(self.__class__)
        self.type = type_

    @classmethod
    def parser(cls, buf):
        esi_type, = struct.unpack_from(cls._PACK_STR, bytes(buf))
        subcls = cls._lookup_type(esi_type)
        return subcls(**subcls.parse_value(buf[1:cls._ESI_LEN]))

    def serialize(self):
        buf = bytearray()
        msg_pack_into(EvpnEsi._PACK_STR, buf, 0, self.type)
        return bytes(buf + self.serialize_value())

    @property
    def formatted_str(self):
        return '%s(%s)' % (self._TYPE_NAME, ','.join((str(getattr(self, v)) for v in self._VALUE_FIELDS)))