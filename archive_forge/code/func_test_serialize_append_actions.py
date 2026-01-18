import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
def test_serialize_append_actions(self):
    c = self._get_obj(True)
    c.serialize()
    self.assertEqual(ofproto.OFP_VERSION, c.version)
    self.assertEqual(ofproto.OFPT_VENDOR, c.msg_type)
    self.assertEqual(0, c.xid)
    self.assertEqual(ofproto_common.NX_EXPERIMENTER_ID, c.vendor)
    fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + ofproto.NICIRA_HEADER_PACK_STR.replace('!', '') + ofproto.NX_FLOW_MOD_PACK_STR.replace('!', '') + ofproto.OFP_ACTION_OUTPUT_PACK_STR.replace('!', '')
    res = struct.unpack(fmt, bytes(c.buf))
    self.assertEqual(ofproto.OFP_VERSION, res[0])
    self.assertEqual(ofproto.OFPT_VENDOR, res[1])
    self.assertEqual(len(c.buf), res[2])
    self.assertEqual(0, res[3])
    self.assertEqual(ofproto_common.NX_EXPERIMENTER_ID, res[4])
    self.assertEqual(ofproto.NXT_FLOW_MOD, res[5])
    self.assertEqual(self.cookie['val'], res[6])
    self.assertEqual(self.command['val'], res[7])
    self.assertEqual(self.idle_timeout['val'], res[8])
    self.assertEqual(self.hard_timeout['val'], res[9])
    self.assertEqual(self.priority['val'], res[10])
    self.assertEqual(self.buffer_id['val'], res[11])
    self.assertEqual(self.out_port['val'], res[12])
    self.assertEqual(self.flags['val'], res[13])
    self.assertEqual(0, res[14])
    self.assertEqual(ofproto.OFPAT_OUTPUT, res[15])
    self.assertEqual(ofproto.OFP_ACTION_OUTPUT_SIZE, res[16])
    self.assertEqual(self.port['val'], res[17])
    self.assertEqual(65509, res[18])