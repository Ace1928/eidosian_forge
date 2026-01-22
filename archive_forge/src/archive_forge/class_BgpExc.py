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
class BgpExc(Exception):
    """Base bgp exception."""
    CODE = 0
    'BGP error code.'
    SUB_CODE = 0
    'BGP error sub-code.'
    SEND_ERROR = True
    'Flag if set indicates Notification message should be sent to peer.'

    def __init__(self, data=''):
        super(BgpExc, self).__init__()
        self.data = data

    def __str__(self):
        return '<%s %r>' % (self.__class__.__name__, self.data)