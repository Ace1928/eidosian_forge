import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import health_monitor
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestHealthMonitorDelete(TestHealthMonitor):

    def setUp(self):
        super().setUp()
        self.cmd = health_monitor.DeleteHealthMonitor(self.app, None)

    def test_health_monitor_delete(self):
        arglist = [self._hm.id]
        verifylist = [('health_monitor', self._hm.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.health_monitor_delete.assert_called_with(health_monitor_id=self._hm.id)

    @mock.patch('osc_lib.utils.wait_for_delete')
    def test_health_monitor_delete_wait(self, mock_wait):
        arglist = [self._hm.id, '--wait']
        verifylist = [('health_monitor', self._hm.id), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.health_monitor_delete.assert_called_with(health_monitor_id=self._hm.id)
        mock_wait.assert_called_once_with(manager=mock.ANY, res_id=self._hm.id, sleep_time=mock.ANY, status_field='provisioning_status')

    def test_health_monitor_delete_failure(self):
        arglist = ['unknown_hm']
        verifylist = [('health_monitor', 'unknown_hm')]
        self.api_mock.health_monitor_list.return_value = {'healthmonitors': []}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertNotCalled(self.api_mock.health_monitor_delete)