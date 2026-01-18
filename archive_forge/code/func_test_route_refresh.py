import logging
import os
import sys
import unittest
from os_ken.utils import binary_str
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import bgp
from os_ken.lib.packet import afi
from os_ken.lib.packet import safi
def test_route_refresh(self):
    msg = bgp.BGPRouteRefresh(afi=afi.IP, safi=safi.MPLS_VPN)
    binmsg = msg.serialize()
    msg2, _, rest = bgp.BGPMessage.parser(binmsg)
    self.assertEqual(str(msg), str(msg2))
    self.assertEqual(len(msg), 23)
    self.assertEqual(rest, b'')