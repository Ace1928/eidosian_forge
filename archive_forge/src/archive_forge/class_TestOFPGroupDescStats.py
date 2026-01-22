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
class TestOFPGroupDescStats(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPGroupDescStats
    """
    length = ofproto.OFP_GROUP_DESC_STATS_SIZE + ofproto.OFP_BUCKET_SIZE + ofproto.OFP_ACTION_OUTPUT_SIZE
    type_ = 128
    group_id = 6606
    port = 10976
    max_len = ofproto.OFP_ACTION_OUTPUT_SIZE
    actions = [OFPActionOutput(port, max_len)]
    buf_actions = bytearray()
    actions[0].serialize(buf_actions, 0)
    weight = 4386
    watch_port = 8006
    watch_group = 3
    buckets = [OFPBucket(weight, watch_port, watch_group, actions)]
    bucket_cnt = 1024

    def test_init(self):
        c = OFPGroupDescStats(self.type_, self.group_id, self.buckets)
        self.assertEqual(self.type_, c.type)
        self.assertEqual(self.group_id, c.group_id)
        self.assertEqual(self.buckets, c.buckets)

    def _test_parser(self, type_, group_id, bucket_cnt):
        length = ofproto.OFP_GROUP_DESC_STATS_SIZE + (ofproto.OFP_BUCKET_SIZE + ofproto.OFP_ACTION_OUTPUT_SIZE) * bucket_cnt
        fmt = ofproto.OFP_GROUP_DESC_STATS_PACK_STR
        buf = pack(fmt, length, type_, group_id)
        buckets = []
        for b in range(bucket_cnt):
            weight = watch_port = watch_group = b
            bucket = OFPBucket(weight, watch_port, watch_group, self.actions)
            buckets.append(bucket)
            buf_buckets = bytearray()
            buckets[b].serialize(buf_buckets, 0)
            buf += bytes(buf_buckets)
        res = OFPGroupDescStats.parser(buf, 0)
        self.assertEqual(type_, res.type)
        self.assertEqual(group_id, res.group_id)
        for b in range(bucket_cnt):
            self.assertEqual(buckets[b].weight, res.buckets[b].weight)
            self.assertEqual(buckets[b].watch_port, res.buckets[b].watch_port)
            self.assertEqual(buckets[b].watch_group, res.buckets[b].watch_group)
            self.assertEqual(buckets[b].actions[0].port, res.buckets[b].actions[0].port)
            self.assertEqual(buckets[b].actions[0].max_len, res.buckets[b].actions[0].max_len)

    def test_parser_mid(self):
        self._test_parser(self.type_, self.group_id, self.bucket_cnt)

    def test_parser_max(self):
        group_id = 4294967295
        type_ = 255
        bucket_cnt = 2047
        self._test_parser(type_, group_id, bucket_cnt)

    def test_parser_min(self):
        group_id = 0
        type_ = ofproto.OFPGT_ALL
        bucket_cnt = 0
        self._test_parser(type_, group_id, bucket_cnt)

    def test_parser_p1(self):
        type_ = ofproto.OFPGT_SELECT
        self._test_parser(type_, self.group_id, self.bucket_cnt)

    def test_parser_p2(self):
        type_ = ofproto.OFPGT_INDIRECT
        self._test_parser(type_, self.group_id, self.bucket_cnt)

    def test_parser_p3(self):
        type_ = ofproto.OFPGT_FF
        self._test_parser(type_, self.group_id, self.bucket_cnt)