import datetime
from unittest import mock
from openstackclient.compute.v2 import usage as usage_cmds
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_usage_list_with_pagination(self):
    arglist = []
    verifylist = [('start', None), ('end', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.projects_mock.list.assert_called_with()
    self.compute_sdk_client.usages.assert_has_calls([mock.call(start=mock.ANY, end=mock.ANY, detailed=True)])
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(tuple(self.data), tuple(data))