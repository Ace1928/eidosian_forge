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
class _FlowSpecBitmask(_FlowSpecOperatorBase):
    """
    Bitmask operator class for Flow Specification NLRI component
    """
    NOT = 1 << 1
    MATCH = 1 << 0
    _comparison_conditions = {'!=': NOT, '==': MATCH}
    _bitmask_flags = {}

    @classmethod
    def _to_value(cls, value):
        try:
            return cls.__dict__[value]
        except KeyError:
            raise ValueError('Invalid params: %s="%s"' % (cls.COMPONENT_NAME, value))

    def to_str(self):
        string = ''
        if self.operator & self.AND:
            string += '&'
        operator = self.operator & (self.NOT | self.MATCH)
        for k, v in self._comparison_conditions.items():
            if operator == v:
                string += k
        plus = ''
        for k, v in self._bitmask_flags.items():
            if self.value & k:
                string += plus + v
                plus = '+'
        return string