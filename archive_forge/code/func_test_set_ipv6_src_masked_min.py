import unittest
import logging
import socket
from struct import *
from os_ken.ofproto.ofproto_v1_2_parser import *
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ether
from os_ken.ofproto.ofproto_parser import MsgBase
from os_ken import utils
from os_ken.lib import addrconv
from os_ken.lib import pack_utils
def test_set_ipv6_src_masked_min(self):
    ipv6 = '2001:db8:bd05:1d2:288a:1fc0:1:10ee'
    mask = '0:0:0:0:0:0:0:0'
    self._test_set_ipv6_src(ipv6, mask)