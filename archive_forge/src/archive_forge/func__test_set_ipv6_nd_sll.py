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
def _test_set_ipv6_nd_sll(self, nd_sll):
    header = ofproto.OXM_OF_IPV6_ND_SLL
    match = OFPMatch()
    nd_sll = mac.haddr_to_bin(nd_sll)
    match.set_ipv6_nd_sll(nd_sll)
    self._test_serialize_and_parser(match, header, nd_sll)