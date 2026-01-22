import binascii
import unittest
import struct
from os_ken import exception
from os_ken.ofproto import ofproto_common, ofproto_parser
from os_ken.ofproto import ofproto_v1_0, ofproto_v1_0_parser
import logging
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