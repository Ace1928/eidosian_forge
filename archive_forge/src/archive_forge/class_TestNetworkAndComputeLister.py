import argparse
from unittest import mock
import openstack
from osc_lib import exceptions
from openstackclient.network import common
from openstackclient.tests.unit import utils
class TestNetworkAndComputeLister(TestNetworkAndCompute):

    def setUp(self):
        super(TestNetworkAndComputeLister, self).setUp()
        self.cmd = FakeNetworkAndComputeLister(self.app, self.namespace)