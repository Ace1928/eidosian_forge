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
def test_parser_single_struct_flase(self):
    version = ofproto.OFP_VERSION
    msg_type = ofproto.OFPT_STATS_REPLY
    msg_len = ofproto.OFP_STATS_REPLY_SIZE + ofproto.OFP_QUEUE_STATS_SIZE
    xid = 2495926989
    fmt = ofproto.OFP_HEADER_PACK_STR
    buf = pack(fmt, version, msg_type, msg_len, xid)
    type_ = ofproto.OFPST_QUEUE
    flags = 11884
    fmt = ofproto.OFP_STATS_REPLY_PACK_STR
    buf += pack(fmt, type_, flags)
    port_no = 41186
    queue_id = 6606
    tx_bytes = 8638420181865882538
    tx_packets = 2856480458895760962
    tx_errors = 6283093430376743019
    body = [OFPQueueStats(port_no, queue_id, tx_bytes, tx_packets, tx_errors)]
    fmt = ofproto.OFP_QUEUE_STATS_PACK_STR
    buf += pack(fmt, port_no, queue_id, tx_bytes, tx_packets, tx_errors)
    res = self.c.parser(object, version, msg_type, msg_len, xid, buf)
    self.assertEqual(version, res.version)
    self.assertEqual(msg_type, res.msg_type)
    self.assertEqual(msg_len, res.msg_len)
    self.assertEqual(xid, res.xid)
    self.assertEqual(type_, res.type)
    self.assertEqual(flags, res.flags)
    self.assertEqual(port_no, res.body[0].port_no)
    self.assertEqual(queue_id, res.body[0].queue_id)
    self.assertEqual(tx_bytes, res.body[0].tx_bytes)
    self.assertEqual(tx_packets, res.body[0].tx_packets)
    self.assertEqual(tx_errors, res.body[0].tx_errors)