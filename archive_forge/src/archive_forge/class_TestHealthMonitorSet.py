import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import health_monitor
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestHealthMonitorSet(TestHealthMonitor):

    def setUp(self):
        super().setUp()
        self.cmd = health_monitor.SetHealthMonitor(self.app, None)

    def test_health_monitor_set(self):
        arglist = [self._hm.id, '--name', 'new_name', '--http-version', str(self._hm.http_version), '--domain-name', self._hm.domain_name]
        verifylist = [('health_monitor', self._hm.id), ('name', 'new_name'), ('http_version', self._hm.http_version), ('domain_name', self._hm.domain_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.health_monitor_set.assert_called_with(self._hm.id, json={'healthmonitor': {'name': 'new_name', 'http_version': self._hm.http_version, 'domain_name': self._hm.domain_name}})

    @mock.patch('osc_lib.utils.wait_for_status')
    def test_health_monitor_set_wait(self, mock_wait):
        arglist = [self._hm.id, '--name', 'new_name', '--wait']
        verifylist = [('health_monitor', self._hm.id), ('name', 'new_name'), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.health_monitor_set.assert_called_with(self._hm.id, json={'healthmonitor': {'name': 'new_name'}})
        mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self._hm.id, sleep_time=mock.ANY, status_field='provisioning_status')

    def test_health_monitor_set_tag(self):
        self.api_mock.health_monitor_show.return_value = {'tags': ['foo']}
        arglist = [self._hm.id, '--tag', 'bar']
        verifylist = [('health_monitor', self._hm.id), ('tags', ['bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.health_monitor_set.assert_called_once()
        kwargs = self.api_mock.health_monitor_set.mock_calls[0][2]
        tags = kwargs['json']['healthmonitor']['tags']
        self.assertEqual(2, len(tags))
        self.assertIn('foo', tags)
        self.assertIn('bar', tags)

    def test_health_monitor_set_tag_no_tag(self):
        self.api_mock.health_monitor_show.return_value = {'tags': ['foo']}
        arglist = [self._hm.id, '--tag', 'bar', '--no-tag']
        verifylist = [('health_monitor', self._hm.id), ('tags', ['bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.health_monitor_set.assert_called_once_with(self._hm.id, json={'healthmonitor': {'tags': ['bar']}})