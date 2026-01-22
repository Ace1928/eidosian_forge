import logging
import unittest
from os_ken.lib.packet.bgp import (
from os_ken.services.protocols.bgp.utils.bgp import create_v4flowspec_actions
from os_ken.services.protocols.bgp.utils.bgp import create_v6flowspec_actions
from os_ken.services.protocols.bgp.utils.bgp import create_l2vpnflowspec_actions
def _test_create_v4flowspec_actions(self, actions, expected_communities):
    communities = create_v4flowspec_actions(actions)
    expected_communities.sort(key=lambda x: x.subtype)
    communities.sort(key=lambda x: x.subtype)
    self.assertEqual(str(expected_communities), str(communities))