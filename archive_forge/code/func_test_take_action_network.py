import argparse
from unittest import mock
import openstack
from osc_lib import exceptions
from openstackclient.network import common
from openstackclient.tests.unit import utils
def test_take_action_network(self):
    arglist = ['common', 'network']
    verifylist = [('common', 'common'), ('network', 'network')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.network_action.assert_called_with(parsed_args)
    self.assertEqual('take_action_network', result)