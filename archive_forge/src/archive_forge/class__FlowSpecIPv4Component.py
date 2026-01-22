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
class _FlowSpecIPv4Component(_FlowSpecComponentBase):
    """
    Base class for Flow Specification for IPv4 NLRI component
    """
    TYPE_DESTINATION_PREFIX = 1
    TYPE_SOURCE_PREFIX = 2
    TYPE_PROTOCOL = 3
    TYPE_PORT = 4
    TYPE_DESTINATION_PORT = 5
    TYPE_SOURCE_PORT = 6
    TYPE_ICMP = 7
    TYPE_ICMP_CODE = 8
    TYPE_TCP_FLAGS = 9
    TYPE_PACKET_LENGTH = 10
    TYPE_DIFFSERV_CODE_POINT = 11
    TYPE_FRAGMENT = 12