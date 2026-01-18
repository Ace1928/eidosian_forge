import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
def test_parser_append_prop(self):
    len_ = {'buf': b'\x00\x10', 'val': ofproto.OFP_QUEUE_PROP_MIN_RATE_SIZE}
    a_type = {'buf': b'\x00\x01', 'val': ofproto.OFPQT_MIN_RATE}
    a_len = {'buf': b'\x00\x10', 'val': ofproto.OFP_QUEUE_PROP_MIN_RATE_SIZE}
    a_zfill0 = b'\x00' * 4
    a_rate = {'buf': b'\x00\x01', 'val': ofproto.OFPQT_MIN_RATE}
    a_zfill1 = b'\x00' * 6
    buf = self.queue_id['buf'] + len_['buf'] + self.zfill + a_type['buf'] + a_len['buf'] + a_zfill0 + a_rate['buf'] + a_zfill1
    res = self.c.parser(buf, 0)
    self.assertEqual(self.queue_id['val'], res.queue_id)
    self.assertEqual(len_['val'], res.len)
    append_cls = res.properties[0]
    self.assertEqual(a_type['val'], append_cls.property)
    self.assertEqual(a_len['val'], append_cls.len)
    self.assertEqual(a_rate['val'], append_cls.rate)