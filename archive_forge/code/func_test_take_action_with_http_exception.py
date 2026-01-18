import argparse
from unittest import mock
import openstack
from osc_lib import exceptions
from openstackclient.network import common
from openstackclient.tests.unit import utils
def test_take_action_with_http_exception(self):
    with mock.patch.object(self.cmd, 'take_action_network') as m_action:
        m_action.side_effect = openstack.exceptions.HttpException('bar')
        self.assertRaisesRegex(exceptions.CommandError, 'bar', self.cmd.take_action, mock.Mock())
    self.app.client_manager.network_endpoint_enabled = False
    with mock.patch.object(self.cmd, 'take_action_compute') as m_action:
        m_action.side_effect = openstack.exceptions.HttpException('bar')
        self.assertRaisesRegex(exceptions.CommandError, 'bar', self.cmd.take_action, mock.Mock())