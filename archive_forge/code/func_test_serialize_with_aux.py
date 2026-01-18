import unittest
import inspect
import logging
from struct import pack, unpack_from, pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet.ipv4 import ipv4
from os_ken.lib.packet.packet import Packet
from os_ken.lib.packet.packet_utils import checksum
from os_ken.lib import addrconv
from os_ken.lib.packet.igmp import igmp
from os_ken.lib.packet.igmp import igmpv3_query
from os_ken.lib.packet.igmp import igmpv3_report
from os_ken.lib.packet.igmp import igmpv3_report_group
from os_ken.lib.packet.igmp import IGMP_TYPE_QUERY
from os_ken.lib.packet.igmp import IGMP_TYPE_REPORT_V3
from os_ken.lib.packet.igmp import MODE_IS_INCLUDE
def test_serialize_with_aux(self):
    self.setUp_with_aux()
    buf = self.g.serialize()
    res = unpack_from(igmpv3_report_group._PACK_STR, bytes(buf))
    aux, = unpack_from('%ds' % (self.aux_len * 4), bytes(buf), igmpv3_report_group._MIN_LEN)
    self.assertEqual(res[0], self.type_)
    self.assertEqual(res[1], self.aux_len)
    self.assertEqual(res[2], self.num)
    self.assertEqual(res[3], addrconv.ipv4.text_to_bin(self.address))
    self.assertEqual(aux, self.aux)