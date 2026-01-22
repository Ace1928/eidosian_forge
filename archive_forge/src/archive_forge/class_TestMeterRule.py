from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_meter_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestMeterRule(network_fakes.TestNetworkV2):

    def setUp(self):
        super(TestMeterRule, self).setUp()
        self.projects_mock = self.app.client_manager.identity.projects
        self.domains_mock = self.app.client_manager.identity.domains