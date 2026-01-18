import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import pool as pool
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('osc_lib.utils.wait_for_delete')
def test_pool_delete_wait(self, mock_wait):
    arglist = [self._po.id, '--wait']
    verifylist = [('pool', self._po.id), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.pool_delete.assert_called_with(pool_id=self._po.id)
    mock_wait.assert_called_once_with(manager=mock.ANY, res_id=self._po.id, sleep_time=mock.ANY, status_field='provisioning_status')