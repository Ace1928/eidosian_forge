from unittest import mock
from magnumclient.osc.v1 import quotas as osc_quotas
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_quotas_update_wrong_args(self):
    arglist = ['--project-id', 'abc', '--resource', 'Cluster', '--hard-limits', '10']
    verifylist = [('project_id', 'abc'), ('resource', 'Cluster'), ('hard_limit', 10)]
    self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)