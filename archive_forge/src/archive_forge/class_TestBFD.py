import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import udp
from os_ken.lib.packet import bfd
from os_ken.lib import addrconv
class TestBFD(unittest.TestCase):

    def setUp(self):
        self.data = b'\xb0\xa8n\x18\xb8\x08d\x87' + b'\x88\xe9\xcb\xc8\x08\x00E\xc0' + b'\x004hI\x00\x00\xff\x11' + b'\xf4s\xac\x1c\x03\x01\xac\x1c' + b'\x03\x02\xc0\x00\x0e\xc8\x00 ' + b'\xd9\x02!\xc0\x03\x18\x00\x00' + b'\x00\x06\x00\x00\x00\x07\x00\x00' + b'\xea`\x00\x00\xea`\x00\x00' + b'\x00\x00'
        self.data_auth_simple = b"\x08\x00'\xd1\x95|\x08\x00" + b"'\xedTA\x08\x00E\xc0" + b'\x00=\x0c\x90\x00\x00\xff\x11' + b'\xbb\x0b\xc0\xa89\x02\xc0\xa8' + b'9\x01\xc0\x00\x0e\xc8\x00)' + b'F5 D\x03!\x00\x00' + b'\x00\x01\x00\x00\x00\x00\x00\x0f' + b'B@\x00\x0fB@\x00\x00' + b'\x00\x00\x01\t\x02sec' + b'ret'
        self.data_auth_md5 = b"\x08\x00'\xd1\x95|\x08\x00" + b"'\xedTA\x08\x00E\xc0" + b'\x00L\x0cD\x00\x00\xff\x11' + b'\xbbH\xc0\xa89\x02\xc0\xa8' + b'9\x01\xc0\x00\x0e\xc8\x008' + b'Q\xbc D\x030\x00\x00' + b'\x00\x01\x00\x00\x00\x00\x00\x0f' + b'B@\x00\x0fB@\x00\x00' + b'\x00\x00\x02\x18\x02\x00\x00\x00' + b'A\xdbf\xa8\xf9%Z\x8b' + b'\xcb~K\xec%\xa6,#' + b'\xda\x0f'
        self.data_auth_sha1 = b"\x08\x00'\xd1\x95|\x08\x00" + b"'\xedTA\x08\x00E\xc0" + b'\x00P\x0b\x90\x00\x00\xff\x11' + b'\xbb\xf8\xc0\xa89\x02\xc0\xa8' + b'9\x01\xc0\x00\x0e\xc8\x00<' + b'\xb9\x92 D\x034\x00\x00' + b'\x00\x01\x00\x00\x00\x00\x00\x0f' + b'B@\x00\x0fB@\x00\x00' + b'\x00\x00\x04\x1c\x02\x00\x00\x00' + b'A\xb1F \x10\x81\x03\xd7' + b'\xf4\xde\x87aL$a\x1f' + b'<\xc1j\x00i#'
        self.auth_keys = {2: b'secret'}

    def tearDown(self):
        pass

    def test_parse(self):
        buf = self.data
        pkt = packet.Packet(buf)
        i = iter(pkt)
        self.assertEqual(type(next(i)), ethernet.ethernet)
        self.assertEqual(type(next(i)), ipv4.ipv4)
        self.assertEqual(type(next(i)), udp.udp)
        self.assertEqual(type(bfd.bfd.parser(next(i))[0]), bfd.bfd)

    def test_parse_with_auth_simple(self):
        buf = self.data_auth_simple
        pkt = packet.Packet(buf)
        i = iter(pkt)
        self.assertEqual(type(next(i)), ethernet.ethernet)
        self.assertEqual(type(next(i)), ipv4.ipv4)
        self.assertEqual(type(next(i)), udp.udp)
        bfd_obj = bfd.bfd.parser(next(i))[0]
        self.assertEqual(type(bfd_obj), bfd.bfd)
        self.assertEqual(type(bfd_obj.auth_cls), bfd.SimplePassword)
        self.assertTrue(bfd_obj.authenticate(self.auth_keys))

    def test_parse_with_auth_md5(self):
        buf = self.data_auth_md5
        pkt = packet.Packet(buf)
        i = iter(pkt)
        self.assertEqual(type(next(i)), ethernet.ethernet)
        self.assertEqual(type(next(i)), ipv4.ipv4)
        self.assertEqual(type(next(i)), udp.udp)
        bfd_obj = bfd.bfd.parser(next(i))[0]
        self.assertEqual(type(bfd_obj), bfd.bfd)
        self.assertEqual(type(bfd_obj.auth_cls), bfd.KeyedMD5)
        self.assertTrue(bfd_obj.authenticate(self.auth_keys))

    def test_parse_with_auth_sha1(self):
        buf = self.data_auth_sha1
        pkt = packet.Packet(buf)
        i = iter(pkt)
        self.assertEqual(type(next(i)), ethernet.ethernet)
        self.assertEqual(type(next(i)), ipv4.ipv4)
        self.assertEqual(type(next(i)), udp.udp)
        bfd_obj = bfd.bfd.parser(next(i))[0]
        self.assertEqual(type(bfd_obj), bfd.bfd)
        self.assertEqual(type(bfd_obj.auth_cls), bfd.KeyedSHA1)
        self.assertTrue(bfd_obj.authenticate(self.auth_keys))

    def test_serialize(self):
        pkt = packet.Packet()
        eth_pkt = ethernet.ethernet('b0:a8:6e:18:b8:08', '64:87:88:e9:cb:c8')
        pkt.add_protocol(eth_pkt)
        ip_pkt = ipv4.ipv4(src='172.28.3.1', dst='172.28.3.2', tos=192, identification=26697, proto=inet.IPPROTO_UDP)
        pkt.add_protocol(ip_pkt)
        udp_pkt = udp.udp(49152, 3784)
        pkt.add_protocol(udp_pkt)
        bfd_pkt = bfd.bfd(ver=1, diag=bfd.BFD_DIAG_CTRL_DETECT_TIME_EXPIRED, state=bfd.BFD_STATE_UP, detect_mult=3, my_discr=6, your_discr=7, desired_min_tx_interval=60000, required_min_rx_interval=60000, required_min_echo_rx_interval=0)
        pkt.add_protocol(bfd_pkt)
        self.assertEqual(len(pkt.protocols), 4)
        pkt.serialize()
        self.assertEqual(pkt.data, self.data)

    def test_serialize_with_auth_simple(self):
        pkt = packet.Packet()
        eth_pkt = ethernet.ethernet('08:00:27:d1:95:7c', '08:00:27:ed:54:41')
        pkt.add_protocol(eth_pkt)
        ip_pkt = ipv4.ipv4(src='192.168.57.2', dst='192.168.57.1', tos=192, identification=3216, proto=inet.IPPROTO_UDP)
        pkt.add_protocol(ip_pkt)
        udp_pkt = udp.udp(49152, 3784)
        pkt.add_protocol(udp_pkt)
        auth_cls = bfd.SimplePassword(auth_key_id=2, password=self.auth_keys[2])
        bfd_pkt = bfd.bfd(ver=1, diag=bfd.BFD_DIAG_NO_DIAG, flags=bfd.BFD_FLAG_AUTH_PRESENT, state=bfd.BFD_STATE_DOWN, detect_mult=3, my_discr=1, your_discr=0, desired_min_tx_interval=1000000, required_min_rx_interval=1000000, required_min_echo_rx_interval=0, auth_cls=auth_cls)
        pkt.add_protocol(bfd_pkt)
        self.assertEqual(len(pkt.protocols), 4)
        pkt.serialize()
        self.assertEqual(pkt.data, self.data_auth_simple)

    def test_serialize_with_auth_md5(self):
        pkt = packet.Packet()
        eth_pkt = ethernet.ethernet('08:00:27:d1:95:7c', '08:00:27:ed:54:41')
        pkt.add_protocol(eth_pkt)
        ip_pkt = ipv4.ipv4(src='192.168.57.2', dst='192.168.57.1', tos=192, identification=3140, proto=inet.IPPROTO_UDP)
        pkt.add_protocol(ip_pkt)
        udp_pkt = udp.udp(49152, 3784)
        pkt.add_protocol(udp_pkt)
        auth_cls = bfd.KeyedMD5(auth_key_id=2, seq=16859, auth_key=self.auth_keys[2])
        bfd_pkt = bfd.bfd(ver=1, diag=bfd.BFD_DIAG_NO_DIAG, flags=bfd.BFD_FLAG_AUTH_PRESENT, state=bfd.BFD_STATE_DOWN, detect_mult=3, my_discr=1, your_discr=0, desired_min_tx_interval=1000000, required_min_rx_interval=1000000, required_min_echo_rx_interval=0, auth_cls=auth_cls)
        pkt.add_protocol(bfd_pkt)
        self.assertEqual(len(pkt.protocols), 4)
        pkt.serialize()
        self.assertEqual(pkt.data, self.data_auth_md5)

    def test_serialize_with_auth_sha1(self):
        pkt = packet.Packet()
        eth_pkt = ethernet.ethernet('08:00:27:d1:95:7c', '08:00:27:ed:54:41')
        pkt.add_protocol(eth_pkt)
        ip_pkt = ipv4.ipv4(src='192.168.57.2', dst='192.168.57.1', tos=192, identification=2960, proto=inet.IPPROTO_UDP)
        pkt.add_protocol(ip_pkt)
        udp_pkt = udp.udp(49152, 3784)
        pkt.add_protocol(udp_pkt)
        auth_cls = bfd.KeyedSHA1(auth_key_id=2, seq=16817, auth_key=self.auth_keys[2])
        bfd_pkt = bfd.bfd(ver=1, diag=bfd.BFD_DIAG_NO_DIAG, flags=bfd.BFD_FLAG_AUTH_PRESENT, state=bfd.BFD_STATE_DOWN, detect_mult=3, my_discr=1, your_discr=0, desired_min_tx_interval=1000000, required_min_rx_interval=1000000, required_min_echo_rx_interval=0, auth_cls=auth_cls)
        pkt.add_protocol(bfd_pkt)
        self.assertEqual(len(pkt.protocols), 4)
        pkt.serialize()
        self.assertEqual(pkt.data, self.data_auth_sha1)

    def test_json(self):
        bfd1 = bfd.bfd(ver=1, diag=bfd.BFD_DIAG_CTRL_DETECT_TIME_EXPIRED, state=bfd.BFD_STATE_UP, detect_mult=3, my_discr=6, your_discr=7, desired_min_tx_interval=60000, required_min_rx_interval=60000, required_min_echo_rx_interval=0)
        jsondict = bfd1.to_jsondict()
        bfd2 = bfd.bfd.from_jsondict(jsondict['bfd'])
        self.assertEqual(str(bfd1), str(bfd2))

    def test_json_with_auth_simple(self):
        auth_cls = bfd.SimplePassword(auth_key_id=2, password=self.auth_keys[2])
        bfd1 = bfd.bfd(ver=1, diag=bfd.BFD_DIAG_NO_DIAG, flags=bfd.BFD_FLAG_AUTH_PRESENT, state=bfd.BFD_STATE_DOWN, detect_mult=3, my_discr=1, your_discr=0, desired_min_tx_interval=1000000, required_min_rx_interval=1000000, required_min_echo_rx_interval=0, auth_cls=auth_cls)
        jsondict = bfd1.to_jsondict()
        bfd2 = bfd.bfd.from_jsondict(jsondict['bfd'])
        self.assertEqual(str(bfd1), str(bfd2))

    def test_json_with_auth_md5(self):
        auth_cls = bfd.KeyedMD5(auth_key_id=2, seq=16859, auth_key=self.auth_keys[2])
        bfd1 = bfd.bfd(ver=1, diag=bfd.BFD_DIAG_NO_DIAG, flags=bfd.BFD_FLAG_AUTH_PRESENT, state=bfd.BFD_STATE_DOWN, detect_mult=3, my_discr=1, your_discr=0, desired_min_tx_interval=1000000, required_min_rx_interval=1000000, required_min_echo_rx_interval=0, auth_cls=auth_cls)
        jsondict = bfd1.to_jsondict()
        bfd2 = bfd.bfd.from_jsondict(jsondict['bfd'])
        self.assertEqual(str(bfd1), str(bfd2))

    def test_json_with_auth_sha1(self):
        auth_cls = bfd.KeyedSHA1(auth_key_id=2, seq=16859, auth_key=self.auth_keys[2])
        bfd1 = bfd.bfd(ver=1, diag=bfd.BFD_DIAG_NO_DIAG, flags=bfd.BFD_FLAG_AUTH_PRESENT, state=bfd.BFD_STATE_DOWN, detect_mult=3, my_discr=1, your_discr=0, desired_min_tx_interval=1000000, required_min_rx_interval=1000000, required_min_echo_rx_interval=0, auth_cls=auth_cls)
        jsondict = bfd1.to_jsondict()
        bfd2 = bfd.bfd.from_jsondict(jsondict['bfd'])
        self.assertEqual(str(bfd1), str(bfd2))