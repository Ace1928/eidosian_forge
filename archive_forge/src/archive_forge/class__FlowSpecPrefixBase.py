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
class _FlowSpecPrefixBase(_FlowSpecIPv4Component, IPAddrPrefix):
    """
    Prefix base class for Flow Specification NLRI component
    """

    def __init__(self, length, addr, type_=None):
        super(_FlowSpecPrefixBase, self).__init__(type_)
        self.length = length
        prefix = '%s/%s' % (addr, length)
        self.addr = str(netaddr.ip.IPNetwork(prefix).network)

    @classmethod
    def parse_body(cls, buf):
        return cls.parser(buf)

    def serialize_body(self):
        return self.serialize()

    @classmethod
    def from_str(cls, value):
        rule = []
        addr, length = value.split('/')
        rule.append(cls(int(length), addr))
        return rule

    @property
    def value(self):
        return '%s/%s' % (self.addr, self.length)

    def to_str(self):
        return self.value