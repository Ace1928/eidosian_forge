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
def test_open2(self):
    opt_param = [bgp.BGPOptParamCapabilityUnknown(cap_code=200, cap_value=b'hoge'), bgp.BGPOptParamCapabilityGracefulRestart(flags=0, time=120, tuples=[]), bgp.BGPOptParamCapabilityRouteRefresh(), bgp.BGPOptParamCapabilityCiscoRouteRefresh(), bgp.BGPOptParamCapabilityMultiprotocol(afi=afi.IP, safi=safi.MPLS_VPN), bgp.BGPOptParamCapabilityCarryingLabelInfo(), bgp.BGPOptParamCapabilityFourOctetAsNumber(as_number=1234567), bgp.BGPOptParamUnknown(type_=99, value=b'fuga')]
    msg = bgp.BGPOpen(my_as=30000, bgp_identifier='192.0.2.2', opt_param=opt_param)
    binmsg = msg.serialize()
    msg2, _, rest = bgp.BGPMessage.parser(binmsg)
    self.assertEqual(str(msg), str(msg2))
    self.assertTrue(len(msg) > 29)
    self.assertEqual(rest, b'')