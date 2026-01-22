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
@_ExtendedCommunity.register_type(_ExtendedCommunity.FLOWSPEC_TRAFFIC_RATE)
class BGPFlowSpecTrafficRateCommunity(_ExtendedCommunity):
    """
    Flow Specification Traffic Filtering Actions for Traffic Rate.

    ========================== ===============================================
    Attribute                  Description
    ========================== ===============================================
    as_number                  Autonomous System number.
    rate_info                  rate information.
    ========================== ===============================================
    """
    _VALUE_PACK_STR = '!BHf'
    _VALUE_FIELDS = ['subtype', 'as_number', 'rate_info']
    ACTION_NAME = 'traffic_rate'

    def __init__(self, **kwargs):
        super(BGPFlowSpecTrafficRateCommunity, self).__init__()
        kwargs['subtype'] = self.SUBTYPE_FLOWSPEC_TRAFFIC_RATE
        self.do_init(BGPFlowSpecTrafficRateCommunity, self, kwargs)

    @classmethod
    def parse_value(cls, buf):
        subtype, as_number, rate_info = struct.unpack_from(cls._VALUE_PACK_STR, buf)
        return {'subtype': subtype, 'as_number': as_number, 'rate_info': rate_info}

    def serialize_value(self):
        return struct.pack(self._VALUE_PACK_STR, self.subtype, self.as_number, self.rate_info)