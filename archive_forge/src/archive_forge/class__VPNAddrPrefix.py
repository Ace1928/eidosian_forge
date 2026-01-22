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
class _VPNAddrPrefix(_AddrPrefix):
    _RD_PACK_STR = '!Q'

    def __init__(self, length, addr, prefixes=(), route_dist=0):
        if isinstance(addr, tuple):
            assert not route_dist
            assert length > struct.calcsize(self._RD_PACK_STR) * 8
            route_dist = addr[0]
            addr = addr[1:]
        else:
            length += struct.calcsize(self._RD_PACK_STR) * 8
        if isinstance(route_dist, str):
            route_dist = _RouteDistinguisher.from_str(route_dist)
        prefixes = prefixes + (route_dist,)
        super(_VPNAddrPrefix, self).__init__(prefixes=prefixes, length=length, addr=addr)

    @classmethod
    def _prefix_to_bin(cls, addr):
        rd = addr[0]
        rest = addr[1:]
        binrd = rd.serialize()
        return binrd + super(_VPNAddrPrefix, cls)._prefix_to_bin(rest)

    @classmethod
    def _prefix_from_bin(cls, binaddr):
        binrd = binaddr[:8]
        binrest = binaddr[8:]
        rd = _RouteDistinguisher.parser(binrd)
        return (rd,) + super(_VPNAddrPrefix, cls)._prefix_from_bin(binrest)