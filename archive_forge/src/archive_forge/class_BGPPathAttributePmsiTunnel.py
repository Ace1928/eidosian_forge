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
@_PathAttribute.register_type(BGP_ATTR_TYEP_PMSI_TUNNEL_ATTRIBUTE)
class BGPPathAttributePmsiTunnel(_PathAttribute):
    """
    P-Multicast Service Interface Tunnel (PMSI Tunnel) attribute
    """
    _VALUE_PACK_STR = '!BB3s'
    _PACK_STR_SIZE = struct.calcsize(_VALUE_PACK_STR)
    _ATTR_FLAGS = BGP_ATTR_FLAG_OPTIONAL | BGP_ATTR_FLAG_TRANSITIVE
    TYPE_NO_TUNNEL_INFORMATION_PRESENT = 0
    TYPE_INGRESS_REPLICATION = 6

    def __init__(self, pmsi_flags, tunnel_type, mpls_label=None, label=None, vni=None, tunnel_id=None, flags=0, type_=None, length=None):
        super(BGPPathAttributePmsiTunnel, self).__init__(flags=flags, type_=type_, length=length)
        self.pmsi_flags = pmsi_flags
        self.tunnel_type = tunnel_type
        self.tunnel_id = tunnel_id
        if label:
            self._label = label
            self._mpls_label, _ = mpls.label_from_bin(label)
            self._vni = vxlan.vni_from_bin(label)
        else:
            self._label = self._serialize_label(mpls_label, vni)
            self._mpls_label = mpls_label
            self._vni = vni

    @classmethod
    def parse_value(cls, buf):
        pmsi_flags, tunnel_type, label = struct.unpack_from(cls._VALUE_PACK_STR, buf)
        value = buf[cls._PACK_STR_SIZE:]
        return {'pmsi_flags': pmsi_flags, 'tunnel_type': tunnel_type, 'label': label, 'tunnel_id': _PmsiTunnelId.parse(tunnel_type, value)}

    def serialize_value(self):
        buf = bytearray()
        msg_pack_into(self._VALUE_PACK_STR, buf, 0, self.pmsi_flags, self.tunnel_type, self._label)
        if self.tunnel_id is not None:
            buf += self.tunnel_id.serialize()
        return buf

    def _serialize_label(self, mpls_label, vni):
        if mpls_label:
            return mpls.label_to_bin(mpls_label, is_bos=True)
        elif vni:
            return vxlan.vni_to_bin(vni)
        else:
            return b'\x00' * 3

    @property
    def mpls_label(self):
        return self._mpls_label

    @mpls_label.setter
    def mpls_label(self, mpls_label):
        self._label = mpls.label_to_bin(mpls_label, is_bos=True)
        self._mpls_label = mpls_label
        self._vni = None

    @property
    def vni(self):
        return self._vni

    @vni.setter
    def vni(self, vni):
        self._label = vxlan.vni_to_bin(vni)
        self._mpls_label = None
        self._vni = vni

    @classmethod
    def from_jsondict(cls, dict_, decode_string=base64.b64decode, **additional_args):
        if isinstance(dict_['tunnel_id'], dict):
            tunnel_id = dict_.pop('tunnel_id')
            ins = super(BGPPathAttributePmsiTunnel, cls).from_jsondict(dict_, decode_string, **additional_args)
            mod = import_module(cls.__module__)
            for key, value in tunnel_id.items():
                tunnel_id_cls = getattr(mod, key)
                ins.tunnel_id = tunnel_id_cls.from_jsondict(value, decode_string, **additional_args)
        else:
            ins = super(BGPPathAttributePmsiTunnel, cls).from_jsondict(dict_, decode_string, **additional_args)
        return ins