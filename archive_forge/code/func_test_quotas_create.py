from unittest import mock
from magnumclient.osc.v1 import quotas as osc_quotas
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_quotas_create(self):
    arglist = ['--project-id', 'abc', '--resource', 'Cluster']
    verifylist = [('project_id', 'abc'), ('resource', 'Cluster')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.quotas_mock.create.assert_called_with(**self._default_args)