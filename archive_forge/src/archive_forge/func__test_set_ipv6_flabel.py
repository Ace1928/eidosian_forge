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
def _test_set_ipv6_flabel(self, flabel, mask=None):
    header = ofproto.OXM_OF_IPV6_FLABEL
    match = OFPMatch()
    if mask is None:
        match.set_ipv6_flabel(flabel)
    else:
        header = ofproto.OXM_OF_IPV6_FLABEL_W
        match.set_ipv6_flabel_masked(flabel, mask)
    self._test_serialize_and_parser(match, header, flabel, mask)