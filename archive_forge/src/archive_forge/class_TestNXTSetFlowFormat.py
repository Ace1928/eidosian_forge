import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestNXTSetFlowFormat(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.NXTSetFlowFormat
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        flow_format = {'buf': b'\xdck\xf5$', 'val': 3698062628}
        c = NXTSetFlowFormat(object, flow_format['val'])
        self.assertEqual(flow_format['val'], c.format)

    def test_parser(self):
        pass

    def test_serialize(self):

        class Datapath(object):
            ofproto = ofproto
            ofproto_parser = ofproto_v1_0_parser
        flow_format = {'buf': b'ZNY\xad', 'val': 1515084205}
        c = NXTSetFlowFormat(Datapath, flow_format['val'])
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_VENDOR, c.msg_type)
        self.assertEqual(0, c.xid)
        self.assertEqual(ofproto_common.NX_EXPERIMENTER_ID, c.vendor)
        fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + ofproto.NICIRA_HEADER_PACK_STR.replace('!', '') + ofproto.NX_SET_FLOW_FORMAT_PACK_STR.replace('!', '')
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(ofproto.OFP_VERSION, res[0])
        self.assertEqual(ofproto.OFPT_VENDOR, res[1])
        self.assertEqual(len(c.buf), res[2])
        self.assertEqual(0, res[3])
        self.assertEqual(ofproto_common.NX_EXPERIMENTER_ID, res[4])
        self.assertEqual(ofproto.NXT_SET_FLOW_FORMAT, res[5])
        self.assertEqual(flow_format['val'], res[6])