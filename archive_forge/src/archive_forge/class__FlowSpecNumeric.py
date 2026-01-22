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
class _FlowSpecNumeric(_FlowSpecOperatorBase):
    """
    Numeric operator class for Flow Specification NLRI component
    """
    LT = 1 << 2
    GT = 1 << 1
    EQ = 1 << 0
    _comparison_conditions = {'==': EQ, '<': LT, '>': GT, '<=': LT | EQ, '>=': GT | EQ}

    @classmethod
    def _to_value(cls, value):
        try:
            return int(str(value), 0)
        except ValueError:
            raise ValueError('Invalid params: %s="%s"' % (cls.COMPONENT_NAME, value))

    def to_str(self):
        string = ''
        if self.operator & self.AND:
            string += '&'
        operator = self.operator & (self.LT | self.GT | self.EQ)
        for k, v in self._comparison_conditions.items():
            if operator == v:
                string += k
        string += str(self.value)
        return string

    @classmethod
    def normalize_operator(cls, operator):
        if operator & (cls.LT | cls.GT | cls.EQ):
            return operator
        else:
            return operator | cls.EQ