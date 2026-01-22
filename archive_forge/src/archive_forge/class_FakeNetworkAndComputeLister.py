import argparse
from unittest import mock
import openstack
from osc_lib import exceptions
from openstackclient.network import common
from openstackclient.tests.unit import utils
class FakeNetworkAndComputeLister(common.NetworkAndComputeLister):

    def update_parser_common(self, parser):
        return _add_common_argument(parser)

    def update_parser_network(self, parser):
        return _add_network_argument(parser)

    def update_parser_compute(self, parser):
        return _add_compute_argument(parser)

    def take_action_network(self, client, parsed_args):
        return client.network_action(parsed_args)

    def take_action_compute(self, client, parsed_args):
        return client.compute_action(parsed_args)