import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPDescStatsReply(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPDescStatsReply
    """

    class Datapath(object):
        ofproto = ofproto
        ofproto_parser = ofproto_v1_0_parser
    c = OFPDescStatsReply(Datapath)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        pass

    def test_parser(self):
        version = {'buf': b'\x01', 'val': ofproto.OFP_VERSION}
        msg_type = {'buf': b'\x11', 'val': ofproto.OFPT_STATS_REPLY}
        msg_len_val = ofproto.OFP_STATS_MSG_SIZE + ofproto.OFP_DESC_STATS_SIZE
        msg_len = {'buf': b'\x048', 'val': msg_len_val}
        xid = {'buf': b'\x94\xc4\xd2\xcd', 'val': 2495926989}
        buf = version['buf'] + msg_type['buf'] + msg_len['buf'] + xid['buf']
        type_ = {'buf': b'\x00\x00', 'val': ofproto.OFPST_DESC}
        flags = {'buf': b'0\xd9', 'val': 12505}
        buf += type_['buf'] + flags['buf']
        mfr_desc = b'mfr_desc'.ljust(256)
        hw_desc = b'hw_desc'.ljust(256)
        sw_desc = b'sw_desc'.ljust(256)
        serial_num = b'serial_num'.ljust(32)
        dp_desc = b'dp_desc'.ljust(256)
        buf += mfr_desc + hw_desc + sw_desc + serial_num + dp_desc
        res = OFPDescStatsReply.parser(object, version['val'], msg_type['val'], msg_len['val'], xid['val'], buf)
        self.assertEqual(version['val'], res.version)
        self.assertEqual(msg_type['val'], res.msg_type)
        self.assertEqual(msg_len['val'], res.msg_len)
        self.assertEqual(xid['val'], res.xid)
        self.assertEqual(type_['val'], res.type)
        self.assertEqual(flags['val'], res.flags)
        body = res.body
        self.assertEqual(mfr_desc, body.mfr_desc)
        self.assertEqual(hw_desc, body.hw_desc)
        self.assertEqual(sw_desc, body.sw_desc)
        self.assertEqual(serial_num, body.serial_num)
        self.assertEqual(dp_desc, body.dp_desc)

    def test_serialize(self):
        pass