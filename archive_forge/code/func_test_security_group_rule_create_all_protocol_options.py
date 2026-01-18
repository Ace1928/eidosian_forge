from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_security_group_rule_create_all_protocol_options(self, sgr_mock):
    arglist = ['--protocol', 'tcp', '--proto', 'tcp', self._security_group['id']]
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])