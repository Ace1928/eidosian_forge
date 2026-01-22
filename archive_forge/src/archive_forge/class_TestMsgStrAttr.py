import binascii
import unittest
import struct
from os_ken import exception
from os_ken.ofproto import ofproto_common, ofproto_parser
from os_ken.ofproto import ofproto_v1_0, ofproto_v1_0_parser
import logging
class TestMsgStrAttr(unittest.TestCase):
    """ Test case for ofproto_parser.msg_str_attr
    """

    def test_msg_str_attr(self):

        class Check(object):
            check = 'msg_str_attr_test'
        c = Check()
        buf = ''
        res = ofproto_parser.msg_str_attr(c, buf, ('check',))
        str_ = str(res)
        str_ = str_.rsplit()
        self.assertEqual('check', str_[0])
        self.assertEqual('msg_str_attr_test', str_[1])