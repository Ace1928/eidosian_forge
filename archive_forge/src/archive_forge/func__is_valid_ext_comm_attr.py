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
def _is_valid_ext_comm_attr(self, attr):
    """Validates *attr* as string representation of RT or SOO.

        Returns True if *attr* is as per our convention of RT or SOO, else
        False. Our convention is to represent RT/SOO is a string with format:
        *global_admin_part:local_admin_path*
        """
    is_valid = True
    if not isinstance(attr, str):
        is_valid = False
    else:
        first, second = attr.split(':')
        try:
            if '.' in first:
                socket.inet_aton(first)
            else:
                int(first)
                int(second)
        except (ValueError, socket.error):
            is_valid = False
    return is_valid