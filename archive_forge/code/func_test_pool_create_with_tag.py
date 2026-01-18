import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import pool as pool
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_pool_attrs')
def test_pool_create_with_tag(self, mock_attrs):
    mock_attrs.return_value = self.pool_info
    arglist = ['--loadbalancer', 'mock_lb_id', '--name', self._po.name, '--protocol', 'HTTP', '--lb-algorithm', 'ROUND_ROBIN', '--tag', 'foo']
    verifylist = [('loadbalancer', 'mock_lb_id'), ('name', self._po.name), ('protocol', 'HTTP'), ('lb_algorithm', 'ROUND_ROBIN'), ('tags', ['foo'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.pool_create.assert_called_with(json={'pool': self.pool_info})