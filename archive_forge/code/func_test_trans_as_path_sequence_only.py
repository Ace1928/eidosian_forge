import logging
import unittest
from unittest import mock
from os_ken.lib.packet import bgp
from os_ken.services.protocols.bgp import peer
@mock.patch.object(peer.Peer, 'is_four_octet_as_number_cap_valid', mock.MagicMock(return_value=False))
def test_trans_as_path_sequence_only(self):
    input_as_path = [[65000, 4000, 400000, 300000, 40001]]
    expected_as_path = [[65000, 4000, 23456, 23456, 40001]]
    expected_as4_path = [[65000, 4000, 400000, 300000, 40001]]
    self._test_trans_as_path(input_as_path, expected_as_path, expected_as4_path)