import logging
import unittest
from unittest import mock
from os_ken.lib.packet import bgp
from os_ken.services.protocols.bgp import peer
@mock.patch.object(peer.Peer, '__init__', mock.MagicMock(return_value=None))
def _test_extract_and_reconstruct_as_path(self, path_attributes, ex_as_path_value, ex_aggregator_as_number, ex_aggregator_addr):
    update_msg = bgp.BGPUpdate(path_attributes=path_attributes)
    _peer = peer.Peer(None, None, None, None, None)
    _peer._extract_and_reconstruct_as_path(update_msg)
    umsg_pattrs = update_msg.pathattr_map
    as_path_attr = umsg_pattrs.get(bgp.BGP_ATTR_TYPE_AS_PATH, None)
    as4_path_attr = umsg_pattrs.get(bgp.BGP_ATTR_TYPE_AS4_PATH, None)
    aggregator_attr = umsg_pattrs.get(bgp.BGP_ATTR_TYPE_AGGREGATOR, None)
    as4_aggregator_attr = umsg_pattrs.get(bgp.BGP_ATTR_TYPE_AS4_AGGREGATOR, None)
    self.assertEqual(ex_as_path_value, as_path_attr.value)
    self.assertEqual(None, as4_path_attr)
    self.assertEqual(ex_aggregator_as_number, aggregator_attr.as_number)
    self.assertEqual(ex_aggregator_addr, aggregator_attr.addr)
    self.assertEqual(None, as4_aggregator_attr)