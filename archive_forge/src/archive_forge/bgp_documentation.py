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

        Returns an instance identified with the given `tunnel_type`.

        `tunnel_type` should be a str type value and corresponding to
        BGP Tunnel Encapsulation Attribute Tunnel Type constants name
        omitting `TUNNEL_TYPE_` prefix.

        Example:
            - `gre` means TUNNEL_TYPE_GRE
            - `vxlan` means TUNNEL_TYPE_VXLAN

        And raises AttributeError when the corresponding Tunnel Type
        is not found to the given `tunnel_type`.
        