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
@_FlowSpecComponentBase.register_type(_FlowSpecIPv4Component.TYPE_ICMP_CODE, addr_family.IP)
@_FlowSpecComponentBase.register_type(_FlowSpecIPv6Component.TYPE_ICMP_CODE, addr_family.IP6)
class FlowSpecIcmpCode(_FlowSpecNumeric):
    """ICMP code Flow Specification NLRI component

    Set the code field of an ICMP packet at value.
    """
    COMPONENT_NAME = 'icmp_code'