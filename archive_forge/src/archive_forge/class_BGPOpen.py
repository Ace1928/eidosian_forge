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
@BGPMessage.register_type(BGP_MSG_OPEN)
class BGPOpen(BGPMessage):
    """BGP-4 OPEN Message encoder/decoder class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    ========================== ===============================================
    Attribute                  Description
    ========================== ===============================================
    marker                     Marker field.  Ignored when encoding.
    len                        Length field.  Ignored when encoding.
    type                       Type field.
    version                    Version field.
    my_as                      My Autonomous System field.
                               2 octet unsigned integer.
    hold_time                  Hold Time field.
                               2 octet unsigned integer.
    bgp_identifier             BGP Identifier field.
                               An IPv4 address.
                               For example, '192.0.2.1'
    opt_param_len              Optional Parameters Length field.
                               Ignored when encoding.
    opt_param                  Optional Parameters field.
                               A list of BGPOptParam instances.
                               The default is [].
    ========================== ===============================================
    """
    _PACK_STR = '!BHH4sB'
    _MIN_LEN = BGPMessage._HDR_LEN + struct.calcsize(_PACK_STR)
    _TYPE = {'ascii': ['bgp_identifier']}

    def __init__(self, my_as, bgp_identifier, type_=BGP_MSG_OPEN, opt_param_len=0, opt_param=None, version=_VERSION, hold_time=0, len_=None, marker=None):
        opt_param = opt_param if opt_param else []
        super(BGPOpen, self).__init__(marker=marker, len_=len_, type_=type_)
        self.version = version
        self.my_as = my_as
        self.bgp_identifier = bgp_identifier
        self.hold_time = hold_time
        self.opt_param_len = opt_param_len
        self.opt_param = opt_param

    @property
    def opt_param_cap_map(self):
        cap_map = {}
        for param in self.opt_param:
            if param.type == BGP_OPT_CAPABILITY:
                cap_map[param.cap_code] = param
        return cap_map

    def get_opt_param_cap(self, cap_code):
        return self.opt_param_cap_map.get(cap_code)

    @classmethod
    def parser(cls, buf):
        version, my_as, hold_time, bgp_identifier, opt_param_len = struct.unpack_from(cls._PACK_STR, bytes(buf))
        rest = buf[struct.calcsize(cls._PACK_STR):]
        binopts = rest[:opt_param_len]
        opt_param = []
        while binopts:
            opt, binopts = _OptParam.parser(binopts)
            opt_param.extend(opt)
        return {'version': version, 'my_as': my_as, 'hold_time': hold_time, 'bgp_identifier': addrconv.ipv4.bin_to_text(bgp_identifier), 'opt_param_len': opt_param_len, 'opt_param': opt_param}

    def serialize_tail(self):
        self.version = _VERSION
        binopts = bytearray()
        for opt in self.opt_param:
            binopts += opt.serialize()
        self.opt_param_len = len(binopts)
        msg = bytearray(struct.pack(self._PACK_STR, self.version, self.my_as, self.hold_time, addrconv.ipv4.text_to_bin(self.bgp_identifier), self.opt_param_len))
        msg += binopts
        return msg