import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import mapping
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_set_rules_wrong_file_path(self):
    arglist = ['--rules', identity_fakes.mapping_rules_file_path, identity_fakes.mapping_id]
    verifylist = [('mapping', identity_fakes.mapping_id), ('rules', identity_fakes.mapping_rules_file_path)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)