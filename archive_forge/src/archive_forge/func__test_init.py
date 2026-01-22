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
def _test_init(self, in_port):
    buffer_id = 4294967295
    data = b'Message'
    out_port = 10976
    actions = [OFPActionOutput(out_port, 0)]
    c = OFPPacketOut(_Datapath, buffer_id, in_port, actions, data)
    self.assertEqual(buffer_id, c.buffer_id)
    self.assertEqual(in_port, c.in_port)
    self.assertEqual(0, c.actions_len)
    self.assertEqual(data, c.data)
    self.assertEqual(actions, c.actions)