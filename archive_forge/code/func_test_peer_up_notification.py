import unittest
from time import time
from os_ken.lib.packet import bmp
from os_ken.lib.packet import bgp
from os_ken.lib.packet import afi
from os_ken.lib.packet import safi
def test_peer_up_notification(self):
    opt_param = [bgp.BGPOptParamCapabilityUnknown(cap_code=200, cap_value=b'hoge'), bgp.BGPOptParamCapabilityRouteRefresh(), bgp.BGPOptParamCapabilityMultiprotocol(afi=afi.IP, safi=safi.MPLS_VPN)]
    open_message = bgp.BGPOpen(my_as=40000, bgp_identifier='192.0.2.2', opt_param=opt_param)
    msg = bmp.BMPPeerUpNotification(local_address='192.0.2.2', local_port=179, remote_port=11089, sent_open_message=open_message, received_open_message=open_message, peer_type=bmp.BMP_PEER_TYPE_GLOBAL, is_post_policy=True, peer_distinguisher=0, peer_address='192.0.2.1', peer_as=30000, peer_bgp_id='192.0.2.1', timestamp=self._time())
    binmsg = msg.serialize()
    msg2, rest = bmp.BMPMessage.parser(binmsg)
    self.assertEqual(msg.to_jsondict(), msg2.to_jsondict())
    self.assertEqual(rest, b'')