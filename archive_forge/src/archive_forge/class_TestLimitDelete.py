import copy
from keystoneauth1.exceptions import http as ksa_exceptions
from osc_lib import exceptions
from openstackclient.identity.v3 import limit
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestLimitDelete(TestLimit):

    def setUp(self):
        super(TestLimitDelete, self).setUp()
        self.cmd = limit.DeleteLimit(self.app, None)

    def test_limit_delete(self):
        self.limit_mock.delete.return_value = None
        arglist = [identity_fakes.limit_id]
        verifylist = [('limit_id', [identity_fakes.limit_id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.limit_mock.delete.assert_called_with(identity_fakes.limit_id)
        self.assertIsNone(result)

    def test_limit_delete_with_exception(self):
        return_value = ksa_exceptions.NotFound()
        self.limit_mock.delete.side_effect = return_value
        arglist = ['fake-limit-id']
        verifylist = [('limit_id', ['fake-limit-id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 1 limits failed to delete.', str(e))