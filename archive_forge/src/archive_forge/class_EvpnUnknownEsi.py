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
@EvpnEsi.register_unknown_type()
class EvpnUnknownEsi(EvpnEsi):
    """
    ESI value for unknown type
    """
    _TYPE_NAME = 'unknown'
    _VALUE_PACK_STR = '!9s'
    _VALUE_FIELDS = ['value']

    def __init__(self, value, type_=None):
        super(EvpnUnknownEsi, self).__init__(type_)
        self.value = value

    @property
    def formatted_str(self):
        return '%s(%s)' % (self._TYPE_NAME, binary_str(self.value))