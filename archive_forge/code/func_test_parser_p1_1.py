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
def test_parser_p1_1(self):
    type_ = ofproto.OFPET_BAD_REQUEST
    code = ofproto.OFPBRC_BAD_TYPE
    data = b'Error Message.'
    self._test_parser(type_, code, data)