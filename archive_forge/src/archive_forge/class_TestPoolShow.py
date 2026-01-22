import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import pool as pool
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestPoolShow(TestPool):

    def setUp(self):
        super().setUp()
        self.api_mock.pool_show.return_value = self.pool_info
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock
        self.cmd = pool.ShowPool(self.app, None)

    def test_pool_show(self):
        arglist = [self._po.id]
        verifylist = [('pool', self._po.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.pool_show.assert_called_with(pool_id=self._po.id)