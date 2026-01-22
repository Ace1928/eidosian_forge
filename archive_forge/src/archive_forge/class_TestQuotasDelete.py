from unittest import mock
from magnumclient.osc.v1 import quotas as osc_quotas
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
class TestQuotasDelete(TestQuotas):

    def setUp(self):
        super(TestQuotasDelete, self).setUp()
        self.quotas_mock.delete = mock.Mock()
        self.quotas_mock.delete.return_value = None
        self.cmd = osc_quotas.DeleteQuotas(self.app, None)

    def test_quotas_delete(self):
        arglist = ['--project-id', 'abc', '--resource', 'Cluster']
        verifylist = [('project_id', 'abc'), ('resource', 'Cluster')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.quotas_mock.delete.assert_called_with('abc', 'Cluster')

    def test_quotas_delete_no_project_id(self):
        arglist = ['--resource', 'Cluster']
        verifylist = [('resource', 'Cluster')]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)

    def test_quotas_delete_no_resource(self):
        arglist = ['--project-id', 'abc']
        verifylist = [('project_id', 'abc')]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)

    def test_quotas_delete_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)

    def test_quotas_delete_wrong_args(self):
        arglist = ['--project-ids', 'abc', '--resource', 'Cluster']
        verifylist = []
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)