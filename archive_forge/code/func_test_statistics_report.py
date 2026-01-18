import unittest
from time import time
from os_ken.lib.packet import bmp
from os_ken.lib.packet import bgp
from os_ken.lib.packet import afi
from os_ken.lib.packet import safi
def test_statistics_report(self):
    stats = [{'type': bmp.BMP_STAT_TYPE_REJECTED, 'value': 100}, {'type': bmp.BMP_STAT_TYPE_DUPLICATE_PREFIX, 'value': 200}, {'type': bmp.BMP_STAT_TYPE_DUPLICATE_WITHDRAW, 'value': 300}, {'type': bmp.BMP_STAT_TYPE_ADJ_RIB_IN, 'value': 100000}, {'type': bmp.BMP_STAT_TYPE_LOC_RIB, 'value': 500000}, {'type': bmp.BMP_STAT_TYPE_ADJ_RIB_OUT, 'value': 95000}, {'type': bmp.BMP_STAT_TYPE_EXPORT_RIB, 'value': 50000}, {'type': bmp.BMP_STAT_TYPE_EXPORT_RIB, 'value': 50000}]
    msg = bmp.BMPStatisticsReport(stats=stats, peer_type=bmp.BMP_PEER_TYPE_GLOBAL, is_post_policy=True, peer_distinguisher=0, peer_address='192.0.2.1', peer_as=30000, peer_bgp_id='192.0.2.1', timestamp=self._time())
    binmsg = msg.serialize()
    msg2, rest = bmp.BMPMessage.parser(binmsg)
    self.assertEqual(msg.to_jsondict(), msg2.to_jsondict())
    self.assertEqual(rest, b'')