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
@EvpnNLRI.register_type(EvpnNLRI.INCLUSIVE_MULTICAST_ETHERNET_TAG)
class EvpnInclusiveMulticastEthernetTagNLRI(EvpnNLRI):
    """
    Inclusive Multicast Ethernet Tag route type specific EVPN NLRI
    """
    ROUTE_TYPE_NAME = 'multicast_etag'
    _PACK_STR = '!8sIB%ds'
    NLRI_PREFIX_FIELDS = ['ethernet_tag_id', 'ip_addr']
    _TYPE = {'ascii': ['route_dist', 'ip_addr']}

    def __init__(self, route_dist, ethernet_tag_id, ip_addr, ip_addr_len=None, type_=None, length=None):
        super(EvpnInclusiveMulticastEthernetTagNLRI, self).__init__(type_, length)
        self.route_dist = route_dist
        self.ethernet_tag_id = ethernet_tag_id
        self.ip_addr_len = ip_addr_len
        self.ip_addr = ip_addr

    @classmethod
    def parse_value(cls, buf):
        route_dist, rest = cls._rd_from_bin(buf)
        ethernet_tag_id, rest = cls._ethernet_tag_id_from_bin(rest)
        ip_addr_len, rest = cls._ip_addr_len_from_bin(rest)
        ip_addr, rest = cls._ip_addr_from_bin(rest, ip_addr_len // 8)
        return {'route_dist': route_dist.formatted_str, 'ethernet_tag_id': ethernet_tag_id, 'ip_addr_len': ip_addr_len, 'ip_addr': ip_addr}

    def serialize_value(self):
        route_dist = _RouteDistinguisher.from_str(self.route_dist)
        ip_addr = self._ip_addr_to_bin(self.ip_addr)
        self.ip_addr_len = len(ip_addr) * 8
        return struct.pack(self._PACK_STR % len(ip_addr), route_dist.serialize(), self.ethernet_tag_id, self.ip_addr_len, ip_addr)