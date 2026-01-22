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
@_ExtendedCommunity.register_type(_ExtendedCommunity.IPV4_ADDRESS_SPECIFIC)
class BGPIPv4AddressSpecificExtendedCommunity(_ExtendedCommunity):
    _VALUE_PACK_STR = '!B4sH'
    _VALUE_FIELDS = ['subtype', 'ipv4_address', 'local_administrator']
    _TYPE = {'ascii': ['ipv4_address']}

    def __init__(self, **kwargs):
        super(BGPIPv4AddressSpecificExtendedCommunity, self).__init__()
        self.do_init(BGPIPv4AddressSpecificExtendedCommunity, self, kwargs)

    @classmethod
    def parse_value(cls, buf):
        d_ = super(BGPIPv4AddressSpecificExtendedCommunity, cls).parse_value(buf)
        d_['ipv4_address'] = addrconv.ipv4.bin_to_text(d_['ipv4_address'])
        return d_

    def serialize_value(self):
        return struct.pack(self._VALUE_PACK_STR, self.subtype, addrconv.ipv4.text_to_bin(self.ipv4_address), self.local_administrator)