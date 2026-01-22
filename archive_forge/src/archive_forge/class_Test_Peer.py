import logging
import unittest
from unittest import mock
from os_ken.lib.packet import bgp
from os_ken.services.protocols.bgp import peer
class Test_Peer(unittest.TestCase):
    """
    Test case for peer.Peer
    """

    @mock.patch.object(peer.Peer, '__init__', mock.MagicMock(return_value=None))
    def _test_construct_as_path_attr(self, input_as_path, input_as4_path, expected_as_path):
        input_as_path_attr = bgp.BGPPathAttributeAsPath(input_as_path)
        input_as4_path_attr = bgp.BGPPathAttributeAs4Path(input_as4_path)
        _peer = peer.Peer(None, None, None, None, None)
        output_as_path_attr = _peer._construct_as_path_attr(input_as_path_attr, input_as4_path_attr)
        self.assertEqual(bgp.BGP_ATTR_TYPE_AS_PATH, output_as_path_attr.type)
        self.assertEqual(expected_as_path, output_as_path_attr.path_seg_list)

    def test_construct_as_path_attr_sequence_only(self):
        input_as_path = [[65000, 4000, 23456, 23456, 40001]]
        input_as4_path = [[400000, 300000, 40001]]
        expected_as_path = [[65000, 4000, 400000, 300000, 40001]]
        self._test_construct_as_path_attr(input_as_path, input_as4_path, expected_as_path)

    def test_construct_as_path_attr_aggregated_as_path_1(self):
        input_as_path = [[65000, 4000], {10, 20, 30}, [23456, 23456, 40001]]
        input_as4_path = [[400000, 300000, 40001]]
        expected_as_path = [[65000, 4000], {10, 20, 30}, [400000, 300000, 40001]]
        self._test_construct_as_path_attr(input_as_path, input_as4_path, expected_as_path)

    def test_construct_as_path_attr_aggregated_as_path_2(self):
        input_as_path = [[65000, 4000], {10, 20, 30}, [23456, 23456, 40001]]
        input_as4_path = [[3000, 400000, 300000, 40001]]
        expected_as_path = [[65000, 4000, 3000, 400000, 300000, 40001]]
        self._test_construct_as_path_attr(input_as_path, input_as4_path, expected_as_path)

    def test_construct_as_path_attr_aggregated_path_3(self):
        input_as_path = [[65000, 4000, 23456, 23456, 40001]]
        input_as4_path = [[400000, 300000, 40001], {10, 20, 30}]
        expected_as_path = [[65000, 400000, 300000, 40001], {10, 20, 30}]
        self._test_construct_as_path_attr(input_as_path, input_as4_path, expected_as_path)

    def test_construct_as_path_attr_aggregated_as4_path(self):
        input_as_path = [[65000, 4000, 23456, 23456, 40001]]
        input_as4_path = [{10, 20, 30}, [400000, 300000, 40001]]
        expected_as_path = [[65000], {10, 20, 30}, [400000, 300000, 40001]]
        self._test_construct_as_path_attr(input_as_path, input_as4_path, expected_as_path)

    def test_construct_as_path_attr_too_short_as_path(self):
        input_as_path = [[65000, 4000, 23456, 23456, 40001]]
        input_as4_path = [[100000, 65000, 4000, 400000, 300000, 40001]]
        expected_as_path = [[65000, 4000, 23456, 23456, 40001]]
        self._test_construct_as_path_attr(input_as_path, input_as4_path, expected_as_path)

    def test_construct_as_path_attr_too_short_as4_path(self):
        input_as_path = [[65000, 4000, 23456, 23456, 40001]]
        input_as4_path = [[300000, 40001]]
        expected_as_path = [[65000, 4000, 23456, 300000, 40001]]
        self._test_construct_as_path_attr(input_as_path, input_as4_path, expected_as_path)

    def test_construct_as_path_attr_empty_as4_path(self):
        input_as_path = [[65000, 4000, 23456, 23456, 40001]]
        input_as4_path = [[]]
        expected_as_path = [[65000, 4000, 23456, 23456, 40001]]
        self._test_construct_as_path_attr(input_as_path, input_as4_path, expected_as_path)

    @mock.patch.object(peer.Peer, '__init__', mock.MagicMock(return_value=None))
    def test_construct_as_path_attr_as4_path_None(self):
        input_as_path = [[65000, 4000, 23456, 23456, 40001]]
        expected_as_path = [[65000, 4000, 23456, 23456, 40001]]
        input_as_path_attr = bgp.BGPPathAttributeAsPath(input_as_path)
        input_as4_path_attr = None
        _peer = peer.Peer(None, None, None, None, None)
        output_as_path_attr = _peer._construct_as_path_attr(input_as_path_attr, input_as4_path_attr)
        self.assertEqual(bgp.BGP_ATTR_TYPE_AS_PATH, output_as_path_attr.type)
        self.assertEqual(expected_as_path, output_as_path_attr.path_seg_list)

    @mock.patch.object(peer.Peer, '__init__', mock.MagicMock(return_value=None))
    def _test_trans_as_path(self, input_as_path, expected_as_path, expected_as4_path):
        _peer = peer.Peer(None, None, None, None, None)
        output_as_path, output_as4_path = _peer._trans_as_path(input_as_path)
        self.assertEqual(expected_as_path, output_as_path)
        self.assertEqual(expected_as4_path, output_as4_path)

    @mock.patch.object(peer.Peer, 'is_four_octet_as_number_cap_valid', mock.MagicMock(return_value=True))
    def test_trans_as_path_as4_path_is_supported(self):
        input_as_path = [[65000, 4000, 400000, 300000, 40001]]
        expected_as_path = [[65000, 4000, 400000, 300000, 40001]]
        expected_as4_path = None
        self._test_trans_as_path(input_as_path, expected_as_path, expected_as4_path)

    @mock.patch.object(peer.Peer, 'is_four_octet_as_number_cap_valid', mock.MagicMock(return_value=False))
    def test_trans_as_path_sequence_only(self):
        input_as_path = [[65000, 4000, 400000, 300000, 40001]]
        expected_as_path = [[65000, 4000, 23456, 23456, 40001]]
        expected_as4_path = [[65000, 4000, 400000, 300000, 40001]]
        self._test_trans_as_path(input_as_path, expected_as_path, expected_as4_path)

    @mock.patch.object(peer.Peer, 'is_four_octet_as_number_cap_valid', mock.MagicMock(return_value=False))
    def test_trans_as_path_no_trans(self):
        input_as_path = [[65000, 4000, 40000, 30000, 40001]]
        expected_as_path = [[65000, 4000, 40000, 30000, 40001]]
        expected_as4_path = None
        self._test_trans_as_path(input_as_path, expected_as_path, expected_as4_path)

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

    @mock.patch.object(peer.Peer, '__init__', mock.MagicMock(return_value=None))
    def test_extract_and_reconstruct_as_path_with_no_as4_attr(self):
        in_as_path_value = [[1000, 2000, 3000]]
        in_aggregator_as_number = 4000
        in_aggregator_addr = '10.0.0.1'
        ex_as_path_value = [[1000, 2000, 3000]]
        ex_aggregator_as_number = 4000
        ex_aggregator_addr = '10.0.0.1'
        path_attributes = [bgp.BGPPathAttributeAsPath(value=in_as_path_value), bgp.BGPPathAttributeAggregator(as_number=in_aggregator_as_number, addr=in_aggregator_addr)]
        self._test_extract_and_reconstruct_as_path(path_attributes, ex_as_path_value, ex_aggregator_as_number, ex_aggregator_addr)

    @mock.patch.object(peer.Peer, '__init__', mock.MagicMock(return_value=None))
    def test_extract_and_reconstruct_as_path_with_as4_attr(self):
        in_as_path_value = [[1000, 23456, 3000]]
        in_as4_path_value = [[2000, 3000]]
        in_aggregator_as_number = 23456
        in_aggregator_addr = '10.0.0.1'
        in_as4_aggregator_as_number = 4000
        in_as4_aggregator_addr = '10.0.0.1'
        ex_as_path_value = [[1000, 2000, 3000]]
        ex_aggregator_as_number = 4000
        ex_aggregator_addr = '10.0.0.1'
        path_attributes = [bgp.BGPPathAttributeAsPath(value=in_as_path_value), bgp.BGPPathAttributeAs4Path(value=in_as4_path_value), bgp.BGPPathAttributeAggregator(as_number=in_aggregator_as_number, addr=in_aggregator_addr), bgp.BGPPathAttributeAs4Aggregator(as_number=in_as4_aggregator_as_number, addr=in_as4_aggregator_addr)]
        self._test_extract_and_reconstruct_as_path(path_attributes, ex_as_path_value, ex_aggregator_as_number, ex_aggregator_addr)

    @mock.patch.object(peer.Peer, '__init__', mock.MagicMock(return_value=None))
    def test_extract_and_reconstruct_as_path_with_not_trans_as_aggr(self):
        in_as_path_value = [[1000, 23456, 3000]]
        in_as4_path_value = [[2000, 3000]]
        in_aggregator_as_number = 4000
        in_aggregator_addr = '10.0.0.1'
        in_as4_aggregator_as_number = 4000
        in_as4_aggregator_addr = '10.0.0.1'
        ex_as_path_value = [[1000, 23456, 3000]]
        ex_aggregator_as_number = 4000
        ex_aggregator_addr = '10.0.0.1'
        path_attributes = [bgp.BGPPathAttributeAsPath(value=in_as_path_value), bgp.BGPPathAttributeAs4Path(value=in_as4_path_value), bgp.BGPPathAttributeAggregator(as_number=in_aggregator_as_number, addr=in_aggregator_addr), bgp.BGPPathAttributeAs4Aggregator(as_number=in_as4_aggregator_as_number, addr=in_as4_aggregator_addr)]
        self._test_extract_and_reconstruct_as_path(path_attributes, ex_as_path_value, ex_aggregator_as_number, ex_aggregator_addr)

    @mock.patch.object(peer.Peer, '__init__', mock.MagicMock(return_value=None))
    def test_extract_and_reconstruct_as_path_with_short_as_path(self):
        in_as_path_value = [[1000, 23456, 3000]]
        in_as4_path_value = [[2000, 3000, 4000, 5000]]
        in_aggregator_as_number = 4000
        in_aggregator_addr = '10.0.0.1'
        ex_as_path_value = [[1000, 23456, 3000]]
        ex_aggregator_as_number = 4000
        ex_aggregator_addr = '10.0.0.1'
        path_attributes = [bgp.BGPPathAttributeAsPath(value=in_as_path_value), bgp.BGPPathAttributeAs4Path(value=in_as4_path_value), bgp.BGPPathAttributeAggregator(as_number=in_aggregator_as_number, addr=in_aggregator_addr)]
        self._test_extract_and_reconstruct_as_path(path_attributes, ex_as_path_value, ex_aggregator_as_number, ex_aggregator_addr)