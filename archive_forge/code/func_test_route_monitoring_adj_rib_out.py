import unittest
from time import time
from os_ken.lib.packet import bmp
from os_ken.lib.packet import bgp
from os_ken.lib.packet import afi
from os_ken.lib.packet import safi
def test_route_monitoring_adj_rib_out(self):
    update = bgp.BGPUpdate()
    msg = bmp.BMPRouteMonitoring(bgp_update=update, peer_type=bmp.BMP_PEER_TYPE_GLOBAL, is_post_policy=True, is_adj_rib_out=True, peer_distinguisher=0, peer_address='192.0.2.1', peer_as=30000, peer_bgp_id='192.0.2.1', timestamp=self._time())
    binmsg = msg.serialize()
    msg2, rest = bmp.BMPMessage.parser(binmsg)
    self.assertEqual(msg.to_jsondict(), msg2.to_jsondict())
    self.assertEqual(rest, b'')