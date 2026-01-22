import binascii
import unittest
import struct
from os_ken import exception
from os_ken.ofproto import ofproto_common, ofproto_parser
from os_ken.ofproto import ofproto_v1_0, ofproto_v1_0_parser
import logging
class TestMsgBase(unittest.TestCase):
    """ Test case for ofproto_parser.MsgBase
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        pass

    def test_set_xid(self):
        xid = 3841413783
        c = ofproto_parser.MsgBase(object)
        c.set_xid(xid)
        self.assertEqual(xid, c.xid)

    def test_set_xid_check_xid(self):
        xid = 2160492514
        c = ofproto_parser.MsgBase(object)
        c.xid = xid
        self.assertRaises(AssertionError, c.set_xid, xid)

    def _test_parser(self, msg_type=ofproto_v1_0.OFPT_HELLO):
        version = ofproto_v1_0.OFP_VERSION
        msg_len = ofproto_v1_0.OFP_HEADER_SIZE
        xid = 2183948390
        data = b'\x00\x01\x02\x03'
        fmt = ofproto_v1_0.OFP_HEADER_PACK_STR
        buf = struct.pack(fmt, version, msg_type, msg_len, xid) + data
        res = ofproto_v1_0_parser.OFPHello.parser(object, version, msg_type, msg_len, xid, bytearray(buf))
        self.assertEqual(version, res.version)
        self.assertEqual(msg_type, res.msg_type)
        self.assertEqual(msg_len, res.msg_len)
        self.assertEqual(xid, res.xid)
        self.assertEqual(bytes(buf), res.buf)
        list_ = ('version', 'msg_type', 'msg_len', 'xid')
        check = {}
        for s in str(res).rsplit(','):
            if '=' in s:
                k, v = s.rsplit('=')
                if k in list_:
                    check[k] = v
        self.assertEqual(hex(ofproto_v1_0.OFP_VERSION), check['version'])
        self.assertEqual(hex(ofproto_v1_0.OFPT_HELLO), check['msg_type'])
        self.assertEqual(hex(msg_len), check['msg_len'])
        self.assertEqual(hex(xid), check['xid'])
        return True

    def test_parser(self):
        self.assertTrue(self._test_parser())

    def test_parser_check_msg_type(self):
        self.assertRaises(AssertionError, self._test_parser, ofproto_v1_0.OFPT_ERROR)

    def _test_serialize(self):

        class Datapath(object):
            ofproto = ofproto_v1_0
            ofproto_parser = ofproto_v1_0_parser
        c = ofproto_v1_0_parser.OFPHello(Datapath)
        c.serialize()
        self.assertEqual(ofproto_v1_0.OFP_VERSION, c.version)
        self.assertEqual(ofproto_v1_0.OFPT_HELLO, c.msg_type)
        self.assertEqual(0, c.xid)
        return True

    def test_serialize(self):
        self.assertTrue(self._test_serialize())