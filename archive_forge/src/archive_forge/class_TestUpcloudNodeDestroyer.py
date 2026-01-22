import sys
import json
from unittest.mock import Mock, call
from libcloud.test import unittest
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.common.upcloud import (
class TestUpcloudNodeDestroyer(unittest.TestCase):

    def setUp(self):
        self.mock_sleep = Mock()
        self.mock_operations = Mock(spec=UpcloudNodeOperations)
        self.destroyer = UpcloudNodeDestroyer(self.mock_operations, sleep_func=self.mock_sleep)

    def test_node_already_in_stopped_state(self):
        self.mock_operations.get_node_state.side_effect = ['stopped']
        self.assertTrue(self.destroyer.destroy_node(1))
        self.assertTrue(self.mock_operations.stop_node.call_count == 0)
        self.mock_operations.destroy_node.assert_called_once_with(1)

    def test_node_in_error_state(self):
        self.mock_operations.get_node_state.side_effect = ['error']
        self.assertFalse(self.destroyer.destroy_node(1))
        self.assertTrue(self.mock_operations.stop_node.call_count == 0)
        self.assertTrue(self.mock_operations.destroy_node.call_count == 0)

    def test_node_in_started_state(self):
        self.mock_operations.get_node_state.side_effect = ['started', 'stopped']
        self.assertTrue(self.destroyer.destroy_node(1))
        self.mock_operations.stop_node.assert_called_once_with(1)
        self.mock_operations.destroy_node.assert_called_once_with(1)

    def test_node_in_maintenace_state(self):
        self.mock_operations.get_node_state.side_effect = ['maintenance', 'maintenance', None]
        self.assertTrue(self.destroyer.destroy_node(1))
        self.mock_sleep.assert_has_calls([call(self.destroyer.WAIT_AMOUNT), call(self.destroyer.WAIT_AMOUNT)])
        self.assertTrue(self.mock_operations.stop_node.call_count == 0)
        self.assertTrue(self.mock_operations.destroy_node.call_count == 0)

    def test_node_statys_in_started_state_for_awhile(self):
        self.mock_operations.get_node_state.side_effect = ['started', 'started', 'stopped']
        self.assertTrue(self.destroyer.destroy_node(1))
        self.mock_operations.stop_node.assert_called_once_with(1)
        self.mock_sleep.assert_has_calls([call(self.destroyer.WAIT_AMOUNT)])
        self.mock_operations.destroy_node.assert_called_once_with(1)

    def test_reuse(self):
        """Verify that internal flag self.destroyer._stop_node is handled properly"""
        self.mock_operations.get_node_state.side_effect = ['started', 'stopped', 'started', 'stopped']
        self.assertTrue(self.destroyer.destroy_node(1))
        self.assertTrue(self.destroyer.destroy_node(1))
        self.assertEqual(self.mock_sleep.call_count, 0)
        self.assertEqual(self.mock_operations.stop_node.call_count, 2)

    def test_timeout(self):
        self.mock_operations.get_node_state.side_effect = ['maintenance'] * 50
        self.assertRaises(UpcloudTimeoutException, self.destroyer.destroy_node, 1)

    def test_timeout_reuse(self):
        """Verify sleep count is handled properly"""
        self.mock_operations.get_node_state.side_effect = ['maintenance'] * 50
        self.assertRaises(UpcloudTimeoutException, self.destroyer.destroy_node, 1)
        self.mock_operations.get_node_state.side_effect = ['maintenance', None]
        self.assertTrue(self.destroyer.destroy_node(1))