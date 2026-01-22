import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7policy
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestL7PolicyCreate(TestL7Policy):

    def setUp(self):
        super().setUp()
        self.api_mock.l7policy_create.return_value = {'l7policy': self.l7po_info}
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock
        self.cmd = l7policy.CreateL7Policy(self.app, None)

    @mock.patch('octaviaclient.osc.v2.utils.get_l7policy_attrs')
    def test_l7policy_create(self, mock_attrs):
        mock_attrs.return_value = {'listener_id': self._l7po.listener_id, 'name': self._l7po.name, 'action': 'REDIRECT_TO_POOL', 'redirect_pool_id': self._l7po.redirect_pool_id}
        arglist = ['mock_li_id', '--name', self._l7po.name, '--action', 'REDIRECT_TO_POOL'.lower(), '--redirect-pool', self._l7po.redirect_pool_id]
        verifylist = [('listener', 'mock_li_id'), ('name', self._l7po.name), ('action', 'REDIRECT_TO_POOL'), ('redirect_pool', self._l7po.redirect_pool_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7policy_create.assert_called_with(json={'l7policy': {'listener_id': self._l7po.listener_id, 'name': self._l7po.name, 'action': 'REDIRECT_TO_POOL', 'redirect_pool_id': self._l7po.redirect_pool_id}})

    @mock.patch('osc_lib.utils.wait_for_status')
    @mock.patch('octaviaclient.osc.v2.utils.get_l7policy_attrs')
    def test_l7policy_create_wait(self, mock_attrs, mock_wait):
        mock_attrs.return_value = {'listener_id': self._l7po.listener_id, 'name': self._l7po.name, 'action': 'REDIRECT_TO_POOL', 'redirect_pool_id': self._l7po.redirect_pool_id}
        self.api_mock.listener_show.return_value = {'loadbalancers': [{'id': 'mock_lb_id'}]}
        self.api_mock.l7policy_show.return_value = self.l7po_info
        arglist = ['mock_li_id', '--name', self._l7po.name, '--action', 'REDIRECT_TO_POOL'.lower(), '--redirect-pool', self._l7po.redirect_pool_id, '--wait']
        verifylist = [('listener', 'mock_li_id'), ('name', self._l7po.name), ('action', 'REDIRECT_TO_POOL'), ('redirect_pool', self._l7po.redirect_pool_id), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7policy_create.assert_called_with(json={'l7policy': {'listener_id': self._l7po.listener_id, 'name': self._l7po.name, 'action': 'REDIRECT_TO_POOL', 'redirect_pool_id': self._l7po.redirect_pool_id}})
        mock_wait.assert_called_once_with(status_f=mock.ANY, res_id='mock_lb_id', sleep_time=mock.ANY, status_field='provisioning_status')

    @mock.patch('octaviaclient.osc.v2.utils.get_l7policy_attrs')
    def test_l7policy_create_with_tag(self, mock_attrs):
        mock_attrs.return_value = {'listener_id': self._l7po.listener_id, 'name': self._l7po.name, 'action': 'REDIRECT_TO_POOL', 'redirect_pool_id': self._l7po.redirect_pool_id, 'tags': ['foo']}
        arglist = ['mock_li_id', '--name', self._l7po.name, '--action', 'REDIRECT_TO_POOL'.lower(), '--redirect-pool', self._l7po.redirect_pool_id, '--tag', 'foo']
        verifylist = [('listener', 'mock_li_id'), ('name', self._l7po.name), ('action', 'REDIRECT_TO_POOL'), ('redirect_pool', self._l7po.redirect_pool_id), ('tags', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7policy_create.assert_called_with(json={'l7policy': {'listener_id': self._l7po.listener_id, 'name': self._l7po.name, 'action': 'REDIRECT_TO_POOL', 'redirect_pool_id': self._l7po.redirect_pool_id, 'tags': ['foo']}})