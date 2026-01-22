import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPVendorStatsReply(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPVendorStatsReply
    """

    class Datapath(object):
        ofproto = ofproto
        ofproto_parser = ofproto_v1_0_parser
    c = OFPVendorStatsReply(Datapath)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        pass

    def test_parser(self):
        version = {'buf': b'\x01', 'val': ofproto.OFP_VERSION}
        msg_type = {'buf': b'\x11', 'val': ofproto.OFPT_STATS_REPLY}
        msg_len = {'buf': b'\x00\x18', 'val': ofproto.OFP_STATS_MSG_SIZE + 12}
        xid = {'buf': b'\x94\xc4\xd2\xcd', 'val': 2495926989}
        buf = version['buf'] + msg_type['buf'] + msg_len['buf'] + xid['buf']
        type_ = {'buf': b'\xff\xff', 'val': ofproto.OFPST_VENDOR}
        flags = {'buf': b'0\xd9', 'val': 12505}
        buf += type_['buf'] + flags['buf']
        specific_data = b'specific_data'
        buf += specific_data
        res = OFPVendorStatsReply.parser(object, version['val'], msg_type['val'], msg_len['val'], xid['val'], buf)
        self.assertEqual(version['val'], res.version)
        self.assertEqual(msg_type['val'], res.msg_type)
        self.assertEqual(msg_len['val'], res.msg_len)
        self.assertEqual(xid['val'], res.xid)
        self.assertEqual(type_['val'], res.type)
        self.assertEqual(flags['val'], res.flags)
        body = res.body[0]
        self.assertEqual(specific_data, body)

    def test_serialize(self):
        pass