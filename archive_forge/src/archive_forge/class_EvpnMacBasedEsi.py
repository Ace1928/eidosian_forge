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
@EvpnEsi.register_type(EvpnEsi.MAC_BASED)
class EvpnMacBasedEsi(EvpnEsi):
    """
    MAC-based ESI Value

    This type indicates a MAC-based ESI Value that
    can be auto-generated or configured by the operator.
    """
    _TYPE_NAME = 'mac_based'
    _VALUE_PACK_STR = '!6s3s'
    _VALUE_FIELDS = ['mac_addr', 'local_disc']
    _TYPE = {'ascii': ['mac_addr']}

    def __init__(self, mac_addr, local_disc, type_=None):
        super(EvpnMacBasedEsi, self).__init__(type_)
        self.mac_addr = mac_addr
        self.local_disc = local_disc

    @classmethod
    def parse_value(cls, buf):
        mac_addr, local_disc = struct.unpack_from(cls._VALUE_PACK_STR, buf)
        return {'mac_addr': addrconv.mac.bin_to_text(mac_addr), 'local_disc': type_desc.Int3.to_user(local_disc)}

    def serialize_value(self):
        return struct.pack(self._VALUE_PACK_STR, addrconv.mac.text_to_bin(self.mac_addr), type_desc.Int3.from_user(self.local_disc))