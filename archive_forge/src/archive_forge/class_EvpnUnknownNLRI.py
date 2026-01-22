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
@EvpnNLRI.register_unknown_type()
class EvpnUnknownNLRI(EvpnNLRI):
    """
    Unknown route type specific EVPN NLRI
    """
    ROUTE_TYPE_NAME = 'unknown'
    NLRI_PREFIX_FIELDS = ['value']

    def __init__(self, value, type_, length=None):
        super(EvpnUnknownNLRI, self).__init__(type_, length)
        self.value = value

    @classmethod
    def parse_value(cls, buf):
        return {'value': buf}

    def serialize_value(self):
        return self.value

    @property
    def formatted_nlri_str(self):
        return '%s(%s)' % (self.ROUTE_TYPE_NAME, binary_str(self.value))