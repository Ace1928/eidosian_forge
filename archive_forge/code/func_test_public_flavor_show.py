from unittest import mock
from openstack.compute.v2 import flavor as _flavor
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import flavor
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_public_flavor_show(self):
    arglist = [self.flavor.name]
    verifylist = [('flavor', self.flavor.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)