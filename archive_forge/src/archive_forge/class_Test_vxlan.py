import logging
import unittest
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import vxlan
class Test_vxlan(unittest.TestCase):
    """
    Test case for VXLAN (RFC 7348) header encoder/decoder class.
    """
    vni = 1193046
    buf = b'\x08\x00\x00\x00\x124V\x00test_payload'
    pkt = vxlan.vxlan(vni)
    jsondict = {'vxlan': {'vni': vni}}

    def test_init(self):
        self.assertEqual(self.vni, self.pkt.vni)

    def test_parser(self):
        parsed_pkt, next_proto_cls, rest_buf = vxlan.vxlan.parser(self.buf)
        self.assertEqual(self.vni, parsed_pkt.vni)
        self.assertEqual(ethernet.ethernet, next_proto_cls)
        self.assertEqual(b'test_payload', rest_buf)

    def test_invalid_flags(self):
        invalid_flags_bug = b'\x00\x00\x00\x00\x124V\x00'
        self.assertRaises(Exception, vxlan.vxlan.parser, invalid_flags_bug)

    def test_serialize(self):
        serialized_buf = self.pkt.serialize(payload=None, prev=None)
        self.assertEqual(self.buf[:vxlan.vxlan._MIN_LEN], serialized_buf)

    def test_from_jsondict(self):
        pkt_from_json = vxlan.vxlan.from_jsondict(self.jsondict[vxlan.vxlan.__name__])
        self.assertEqual(self.vni, pkt_from_json.vni)

    def test_to_jsondict(self):
        jsondict_from_pkt = self.pkt.to_jsondict()
        self.assertEqual(self.jsondict, jsondict_from_pkt)

    def test_vni_from_bin(self):
        vni = vxlan.vni_from_bin(b'\x124V')
        self.assertEqual(self.vni, vni)

    def test_vni_to_bin(self):
        self.assertEqual(b'\x124V', vxlan.vni_to_bin(self.vni))