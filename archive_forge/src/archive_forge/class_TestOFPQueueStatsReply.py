import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPQueueStatsReply(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPQueueStatsReply
    """

    class Datapath(object):
        ofproto = ofproto
        ofproto_parser = ofproto_v1_0_parser
    c = OFPQueueStatsReply(Datapath)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        pass

    def test_parser(self):
        version = {'buf': b'\x01', 'val': ofproto.OFP_VERSION}
        msg_type = {'buf': b'\x11', 'val': ofproto.OFPT_STATS_REPLY}
        msg_len_val = ofproto.OFP_STATS_MSG_SIZE + ofproto.OFP_QUEUE_STATS_SIZE
        msg_len = {'buf': b'\x00,', 'val': msg_len_val}
        xid = {'buf': b'\x19\xfc(l', 'val': 435955820}
        buf = version['buf'] + msg_type['buf'] + msg_len['buf'] + xid['buf']
        type_ = {'buf': b'\x00\x05', 'val': ofproto.OFPST_QUEUE}
        flags = {'buf': b';+', 'val': 15147}
        buf += type_['buf'] + flags['buf']
        port_no = {'buf': b'\xe7k', 'val': 59243}
        zfill = b'\x00' * 2
        queue_id = {'buf': b'*\xa8\x7f2', 'val': 715685682}
        tx_bytes = {'buf': b'w\xe1\xd5c\x18\xaec\xaa', 'val': 8638420181865882538}
        tx_packets = {'buf': b"'\xa4A\xd7\xd4S\x9eB", 'val': 2856480458895760962}
        tx_errors = {'buf': b'W2\x08/\x882@k', 'val': 6283093430376743019}
        buf += port_no['buf'] + zfill + queue_id['buf'] + tx_bytes['buf'] + tx_packets['buf'] + tx_errors['buf']
        res = OFPQueueStatsReply.parser(object, version['val'], msg_type['val'], msg_len['val'], xid['val'], buf)
        self.assertEqual(version['val'], res.version)
        self.assertEqual(msg_type['val'], res.msg_type)
        self.assertEqual(msg_len['val'], res.msg_len)
        self.assertEqual(xid['val'], res.xid)
        self.assertEqual(type_['val'], res.type)
        self.assertEqual(flags['val'], res.flags)
        body = res.body[0]
        self.assertEqual(port_no['val'], body.port_no)
        self.assertEqual(queue_id['val'], body.queue_id)
        self.assertEqual(tx_bytes['val'], body.tx_bytes)
        self.assertEqual(tx_packets['val'], body.tx_packets)
        self.assertEqual(tx_errors['val'], body.tx_errors)

    def test_serialize(self):
        pass