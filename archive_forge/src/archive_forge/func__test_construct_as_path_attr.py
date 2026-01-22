import logging
import unittest
from unittest import mock
from os_ken.lib.packet import bgp
from os_ken.services.protocols.bgp import peer
@mock.patch.object(peer.Peer, '__init__', mock.MagicMock(return_value=None))
def _test_construct_as_path_attr(self, input_as_path, input_as4_path, expected_as_path):
    input_as_path_attr = bgp.BGPPathAttributeAsPath(input_as_path)
    input_as4_path_attr = bgp.BGPPathAttributeAs4Path(input_as4_path)
    _peer = peer.Peer(None, None, None, None, None)
    output_as_path_attr = _peer._construct_as_path_attr(input_as_path_attr, input_as4_path_attr)
    self.assertEqual(bgp.BGP_ATTR_TYPE_AS_PATH, output_as_path_attr.type)
    self.assertEqual(expected_as_path, output_as_path_attr.path_seg_list)