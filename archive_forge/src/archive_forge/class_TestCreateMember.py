import copy
from unittest import mock
import osc_lib.tests.utils as osc_test_utils
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import member
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestCreateMember(TestMember):

    def setUp(self):
        super().setUp()
        self.cmd = member.CreateMember(self.app, None)
        self.api_mock.member_create.return_value = {'member': self.mem_info}

    @mock.patch('octaviaclient.osc.v2.utils.get_member_attrs')
    def test_member_create(self, mock_attrs):
        mock_attrs.return_value = {'ip_address': '192.0.2.122', 'protocol_port': self._mem.protocol_port, 'weight': self._mem.weight, 'admin_state_up': True, 'pool_id': self._mem.pool_id, 'backup': False}
        arglist = ['pool_id', '--address', '192.0.2.122', '--protocol-port', '80', '--weight', '1', '--enable', '--disable-backup']
        verifylist = [('address', '192.0.2.122'), ('protocol_port', 80), ('weight', 1)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.member_create.assert_called_with(pool_id=self._mem.pool_id, json={'member': {'ip_address': '192.0.2.122', 'protocol_port': self._mem.protocol_port, 'weight': self._mem.weight, 'admin_state_up': True, 'backup': False}})

    @mock.patch('osc_lib.utils.wait_for_status')
    @mock.patch('octaviaclient.osc.v2.utils.get_member_attrs')
    def test_member_create_wait(self, mock_attrs, mock_wait):
        mock_attrs.return_value = {'ip_address': '192.0.2.122', 'protocol_port': self._mem.protocol_port, 'weight': self._mem.weight, 'admin_state_up': True, 'pool_id': self._mem.pool_id, 'backup': False}
        self.api_mock.pool_show.return_value = {'loadbalancers': [{'id': 'mock_lb_id'}]}
        self.api_mock.member_show.return_value = self.mem_info
        arglist = ['pool_id', '--address', '192.0.2.122', '--protocol-port', '80', '--weight', '1', '--enable', '--disable-backup', '--wait']
        verifylist = [('address', '192.0.2.122'), ('protocol_port', 80), ('weight', 1), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.member_create.assert_called_with(pool_id=self._mem.pool_id, json={'member': {'ip_address': '192.0.2.122', 'protocol_port': self._mem.protocol_port, 'weight': self._mem.weight, 'admin_state_up': True, 'backup': False}})
        mock_wait.assert_called_once_with(status_f=mock.ANY, res_id='mock_lb_id', sleep_time=mock.ANY, status_field='provisioning_status')

    @mock.patch('octaviaclient.osc.v2.utils.get_member_attrs')
    def test_member_create_with_tag(self, mock_attrs):
        mock_attrs.return_value = {'ip_address': '192.0.2.122', 'protocol_port': self._mem.protocol_port, 'pool_id': self._mem.pool_id, 'tags': ['foo']}
        arglist = ['pool_id', '--address', '192.0.2.122', '--protocol-port', '80', '--tag', 'foo']
        verifylist = [('address', '192.0.2.122'), ('protocol_port', 80), ('tags', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.member_create.assert_called_with(pool_id=self._mem.pool_id, json={'member': {'ip_address': '192.0.2.122', 'protocol_port': self._mem.protocol_port, 'tags': ['foo']}})