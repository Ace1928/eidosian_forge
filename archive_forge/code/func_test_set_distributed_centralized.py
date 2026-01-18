from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_distributed_centralized(self):
    arglist = [self._router.name, '--distributed', '--centralized']
    verifylist = [('router', self._router.name), ('distributed', True), ('distributed', False)]
    self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)