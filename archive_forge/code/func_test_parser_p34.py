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
def test_parser_p34(self):
    self._test_parser_p(ofproto.OFPXMT_OFB_IPV6_ND_TLL, ofproto.OFPIT_EXPERIMENTER, ofproto.OFPTC_TABLE_MISS_MASK)