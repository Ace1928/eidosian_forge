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
def test_serialize_p16(self):
    port_no = ofproto.OFPP_LOCAL
    hw_addr = self.hw_addr
    config = ofproto.OFPPC_NO_PACKET_IN
    mask = ofproto.OFPPC_NO_PACKET_IN
    advertise = ofproto.OFPPF_PAUSE_ASYM
    self._test_serialize(port_no, hw_addr, config, mask, advertise)