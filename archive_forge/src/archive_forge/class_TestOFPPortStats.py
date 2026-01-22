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
class TestOFPPortStats(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPPortStats
    """

    def test_init(self):
        port_no = 6606
        rx_packets = 5999980397101236279
        tx_packets = 2856480458895760962
        rx_bytes = 6170274950576278921
        tx_bytes = 8638420181865882538
        rx_dropped = 6982303461569875546
        tx_dropped = 661287462113808071
        rx_errors = 3422231811478788365
        tx_errors = 6283093430376743019
        rx_frame_err = 876072919806406283
        rx_over_err = 6525873760178941600
        rx_crc_err = 8303073210207070535
        collisions = 3409801584220270201
        res = OFPPortStats(port_no, rx_packets, tx_packets, rx_bytes, tx_bytes, rx_dropped, tx_dropped, rx_errors, tx_errors, rx_frame_err, rx_over_err, rx_crc_err, collisions)
        self.assertEqual(port_no, res.port_no)
        self.assertEqual(rx_packets, res.rx_packets)
        self.assertEqual(tx_packets, res.tx_packets)
        self.assertEqual(rx_bytes, res.rx_bytes)
        self.assertEqual(tx_bytes, res.tx_bytes)
        self.assertEqual(rx_dropped, res.rx_dropped)
        self.assertEqual(tx_dropped, res.tx_dropped)
        self.assertEqual(rx_errors, res.rx_errors)
        self.assertEqual(tx_errors, res.tx_errors)
        self.assertEqual(rx_frame_err, res.rx_frame_err)
        self.assertEqual(rx_over_err, res.rx_over_err)
        self.assertEqual(rx_crc_err, res.rx_crc_err)
        self.assertEqual(collisions, res.collisions)

    def _test_parser(self, port_no, rx_packets, tx_packets, rx_bytes, tx_bytes, rx_dropped, tx_dropped, rx_errors, tx_errors, rx_frame_err, rx_over_err, rx_crc_err, collisions):
        fmt = ofproto.OFP_PORT_STATS_PACK_STR
        buf = pack(fmt, port_no, rx_packets, tx_packets, rx_bytes, tx_bytes, rx_dropped, tx_dropped, rx_errors, tx_errors, rx_frame_err, rx_over_err, rx_crc_err, collisions)
        res = OFPPortStats.parser(buf, 0)
        self.assertEqual(port_no, res.port_no)
        self.assertEqual(rx_packets, res.rx_packets)
        self.assertEqual(tx_packets, res.tx_packets)
        self.assertEqual(rx_bytes, res.rx_bytes)
        self.assertEqual(tx_bytes, res.tx_bytes)
        self.assertEqual(rx_dropped, res.rx_dropped)
        self.assertEqual(tx_dropped, res.tx_dropped)
        self.assertEqual(rx_errors, res.rx_errors)
        self.assertEqual(tx_errors, res.tx_errors)
        self.assertEqual(rx_frame_err, res.rx_frame_err)
        self.assertEqual(rx_over_err, res.rx_over_err)
        self.assertEqual(rx_crc_err, res.rx_crc_err)
        self.assertEqual(collisions, res.collisions)

    def test_parser_mid(self):
        port_no = 6606
        rx_packets = 5999980397101236279
        tx_packets = 2856480458895760962
        rx_bytes = 6170274950576278921
        tx_bytes = 8638420181865882538
        rx_dropped = 6982303461569875546
        tx_dropped = 661287462113808071
        rx_errors = 3422231811478788365
        tx_errors = 6283093430376743019
        rx_frame_err = 876072919806406283
        rx_over_err = 6525873760178941600
        rx_crc_err = 8303073210207070535
        collisions = 3409801584220270201
        self._test_parser(port_no, rx_packets, tx_packets, rx_bytes, tx_bytes, rx_dropped, tx_dropped, rx_errors, tx_errors, rx_frame_err, rx_over_err, rx_crc_err, collisions)

    def test_parser_max(self):
        port_no = 4294967295
        rx_packets = 18446744073709551615
        tx_packets = 18446744073709551615
        rx_bytes = 18446744073709551615
        tx_bytes = 18446744073709551615
        rx_dropped = 18446744073709551615
        tx_dropped = 18446744073709551615
        rx_errors = 18446744073709551615
        tx_errors = 18446744073709551615
        rx_frame_err = 18446744073709551615
        rx_over_err = 18446744073709551615
        rx_crc_err = 18446744073709551615
        collisions = 18446744073709551615
        self._test_parser(port_no, rx_packets, tx_packets, rx_bytes, tx_bytes, rx_dropped, tx_dropped, rx_errors, tx_errors, rx_frame_err, rx_over_err, rx_crc_err, collisions)

    def test_parser_min(self):
        port_no = 0
        rx_packets = 0
        tx_packets = 0
        rx_bytes = 0
        tx_bytes = 0
        rx_dropped = 0
        tx_dropped = 0
        rx_errors = 0
        tx_errors = 0
        rx_frame_err = 0
        rx_over_err = 0
        rx_crc_err = 0
        collisions = 0
        self._test_parser(port_no, rx_packets, tx_packets, rx_bytes, tx_bytes, rx_dropped, tx_dropped, rx_errors, tx_errors, rx_frame_err, rx_over_err, rx_crc_err, collisions)

    def _test_parser_p(self, port_no):
        port_no = port_no
        rx_packets = 5999980397101236279
        tx_packets = 2856480458895760962
        rx_bytes = 6170274950576278921
        tx_bytes = 8638420181865882538
        rx_dropped = 6982303461569875546
        tx_dropped = 661287462113808071
        rx_errors = 3422231811478788365
        tx_errors = 6283093430376743019
        rx_frame_err = 876072919806406283
        rx_over_err = 6525873760178941600
        rx_crc_err = 8303073210207070535
        collisions = 3409801584220270201
        self._test_parser(port_no, rx_packets, tx_packets, rx_bytes, tx_bytes, rx_dropped, tx_dropped, rx_errors, tx_errors, rx_frame_err, rx_over_err, rx_crc_err, collisions)

    def test_parser_p1(self):
        self._test_parser_p(ofproto.OFPP_MAX)

    def test_parser_p2(self):
        self._test_parser_p(ofproto.OFPP_IN_PORT)

    def test_parser_p3(self):
        self._test_parser_p(ofproto.OFPP_TABLE)

    def test_parser_p4(self):
        self._test_parser_p(ofproto.OFPP_NORMAL)

    def test_parser_p5(self):
        self._test_parser_p(ofproto.OFPP_FLOOD)

    def test_parser_p6(self):
        self._test_parser_p(ofproto.OFPP_ALL)

    def test_parser_p7(self):
        self._test_parser_p(ofproto.OFPP_CONTROLLER)

    def test_parser_p8(self):
        self._test_parser_p(ofproto.OFPP_LOCAL)