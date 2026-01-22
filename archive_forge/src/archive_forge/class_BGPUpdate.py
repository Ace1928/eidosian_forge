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
@BGPMessage.register_type(BGP_MSG_UPDATE)
class BGPUpdate(BGPMessage):
    """BGP-4 UPDATE Message encoder/decoder class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ========================== ===============================================
    Attribute                  Description
    ========================== ===============================================
    marker                     Marker field.  Ignored when encoding.
    len                        Length field.  Ignored when encoding.
    type                       Type field.
    withdrawn_routes_len       Withdrawn Routes Length field.
                               Ignored when encoding.
    withdrawn_routes           Withdrawn Routes field.
                               A list of BGPWithdrawnRoute instances.
                               The default is [].
    total_path_attribute_len   Total Path Attribute Length field.
                               Ignored when encoding.
    path_attributes            Path Attributes field.
                               A list of BGPPathAttribute instances.
                               The default is [].
    nlri                       Network Layer Reachability Information field.
                               A list of BGPNLRI instances.
                               The default is [].
    ========================== ===============================================
    """
    _MIN_LEN = BGPMessage._HDR_LEN

    def __init__(self, type_=BGP_MSG_UPDATE, withdrawn_routes_len=None, withdrawn_routes=None, total_path_attribute_len=None, path_attributes=None, nlri=None, len_=None, marker=None):
        withdrawn_routes = withdrawn_routes if withdrawn_routes else []
        path_attributes = path_attributes if path_attributes else []
        nlri = nlri if nlri else []
        super(BGPUpdate, self).__init__(marker=marker, len_=len_, type_=type_)
        self.withdrawn_routes_len = withdrawn_routes_len
        self.withdrawn_routes = withdrawn_routes
        self.total_path_attribute_len = total_path_attribute_len
        self.path_attributes = path_attributes
        self.nlri = nlri

    @property
    def pathattr_map(self):
        passattr_map = {}
        for attr in self.path_attributes:
            passattr_map[attr.type] = attr
        return passattr_map

    def get_path_attr(self, attr_name):
        return self.pathattr_map.get(attr_name)

    @classmethod
    def parser(cls, buf):
        offset = 0
        buf = bytes(buf)
        withdrawn_routes_len, = struct.unpack_from('!H', buf, offset)
        binroutes = buf[offset + 2:offset + 2 + withdrawn_routes_len]
        offset += 2 + withdrawn_routes_len
        total_path_attribute_len, = struct.unpack_from('!H', buf, offset)
        binpathattrs = buf[offset + 2:offset + 2 + total_path_attribute_len]
        binnlri = buf[offset + 2 + total_path_attribute_len:]
        withdrawn_routes = []
        while binroutes:
            r, binroutes = BGPWithdrawnRoute.parser(binroutes)
            withdrawn_routes.append(r)
        path_attributes = []
        while binpathattrs:
            pa, binpathattrs = _PathAttribute.parser(binpathattrs)
            path_attributes.append(pa)
        offset += 2 + total_path_attribute_len
        nlri = []
        while binnlri:
            n, binnlri = BGPNLRI.parser(binnlri)
            nlri.append(n)
        return {'withdrawn_routes_len': withdrawn_routes_len, 'withdrawn_routes': withdrawn_routes, 'total_path_attribute_len': total_path_attribute_len, 'path_attributes': path_attributes, 'nlri': nlri}

    def serialize_tail(self):
        binroutes = bytearray()
        for r in self.withdrawn_routes:
            binroutes += r.serialize()
        self.withdrawn_routes_len = len(binroutes)
        binpathattrs = bytearray()
        for pa in self.path_attributes:
            binpathattrs += pa.serialize()
        self.total_path_attribute_len = len(binpathattrs)
        binnlri = bytearray()
        for n in self.nlri:
            binnlri += n.serialize()
        msg = bytearray()
        offset = 0
        msg_pack_into('!H', msg, offset, self.withdrawn_routes_len)
        msg += binroutes
        offset += 2 + self.withdrawn_routes_len
        msg_pack_into('!H', msg, offset, self.total_path_attribute_len)
        msg += binpathattrs
        offset += 2 + self.total_path_attribute_len
        msg += binnlri
        return msg