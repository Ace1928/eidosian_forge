import copy
from unittest import mock
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7rule
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestL7RuleCreate(TestL7Rule):

    def setUp(self):
        super().setUp()
        self.api_mock.l7rule_create.return_value = {'rule': self.l7rule_info}
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock
        self.cmd = l7rule.CreateL7Rule(self.app, None)

    @mock.patch('octaviaclient.osc.v2.utils.get_l7rule_attrs')
    def test_l7rule_create(self, mock_attrs):
        mock_attrs.return_value = {'l7policy_id': self._l7po.id, 'compare-type': 'ENDS_WITH', 'value': '.example.com', 'type': 'HOST_NAME'}
        arglist = [self._l7po.id, '--compare-type', 'ENDS_WITH', '--value', '.example.com', '--type', 'HOST_NAME'.lower()]
        verifylist = [('l7policy', self._l7po.id), ('compare_type', 'ENDS_WITH'), ('value', '.example.com'), ('type', 'HOST_NAME')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_create.assert_called_with(l7policy_id=self._l7po.id, json={'rule': {'compare-type': 'ENDS_WITH', 'value': '.example.com', 'type': 'HOST_NAME'}})

    @mock.patch('osc_lib.utils.wait_for_status')
    @mock.patch('octaviaclient.osc.v2.utils.get_l7rule_attrs')
    def test_l7rule_create_wait(self, mock_attrs, mock_wait):
        mock_attrs.return_value = {'l7policy_id': self._l7po.id, 'compare-type': 'ENDS_WITH', 'value': '.example.com', 'type': 'HOST_NAME'}
        self.api_mock.l7policy_show.return_value = {'listener_id': 'mock_listener_id'}
        self.api_mock.listener_show.return_value = {'loadbalancers': [{'id': 'mock_lb_id'}]}
        self.api_mock.l7rule_show.return_value = self.l7rule_info
        arglist = [self._l7po.id, '--compare-type', 'ENDS_WITH', '--value', '.example.com', '--type', 'HOST_NAME'.lower(), '--wait']
        verifylist = [('l7policy', self._l7po.id), ('compare_type', 'ENDS_WITH'), ('value', '.example.com'), ('type', 'HOST_NAME'), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_create.assert_called_with(l7policy_id=self._l7po.id, json={'rule': {'compare-type': 'ENDS_WITH', 'value': '.example.com', 'type': 'HOST_NAME'}})
        mock_wait.assert_called_once_with(status_f=mock.ANY, res_id='mock_lb_id', sleep_time=mock.ANY, status_field='provisioning_status')

    @mock.patch('octaviaclient.osc.v2.utils.get_l7rule_attrs')
    def test_l7rule_create_with_tag(self, mock_attrs):
        mock_attrs.return_value = {'l7policy_id': self._l7po.id, 'compare-type': 'ENDS_WITH', 'value': '.example.com', 'type': 'HOST_NAME', 'tags': ['foo']}
        arglist = [self._l7po.id, '--compare-type', 'ENDS_WITH', '--value', '.example.com', '--type', 'HOST_NAME'.lower(), '--tag', 'foo']
        verifylist = [('l7policy', self._l7po.id), ('compare_type', 'ENDS_WITH'), ('value', '.example.com'), ('type', 'HOST_NAME'), ('tags', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.l7rule_create.assert_called_with(l7policy_id=self._l7po.id, json={'rule': {'compare-type': 'ENDS_WITH', 'value': '.example.com', 'type': 'HOST_NAME', 'tags': ['foo']}})