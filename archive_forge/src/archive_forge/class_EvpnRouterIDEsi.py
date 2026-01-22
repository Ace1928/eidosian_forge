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
@EvpnEsi.register_type(EvpnEsi.ROUTER_ID)
class EvpnRouterIDEsi(EvpnEsi):
    """
    Router-ID ESI Value

    This type indicates a router-ID ESI Value that
    can be auto-generated or configured by the operator.
    """
    _TYPE_NAME = 'router_id'
    _VALUE_PACK_STR = '!4sIx'
    _VALUE_FIELDS = ['router_id', 'local_disc']
    _TYPE = {'ascii': ['router_id']}

    def __init__(self, router_id, local_disc, type_=None):
        super(EvpnRouterIDEsi, self).__init__(type_)
        self.router_id = router_id
        self.local_disc = local_disc

    @classmethod
    def parse_value(cls, buf):
        router_id, local_disc = struct.unpack_from(cls._VALUE_PACK_STR, buf)
        return {'router_id': addrconv.ipv4.bin_to_text(router_id), 'local_disc': local_disc}

    def serialize_value(self):
        return struct.pack(self._VALUE_PACK_STR, addrconv.ipv4.text_to_bin(self.router_id), self.local_disc)