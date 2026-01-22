import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.logging import network_log
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.logging import fakes
class TestListNetworkLog(TestNetworkLog):

    def _setup_summary(self, expect=None):
        event = 'Event: ' + self.res['event'].upper()
        target = 'Logged: (None specified)'
        if expect:
            if expect.get('event'):
                event = expect['event']
            if expect.get('resource'):
                target = expect['resource']
        summary = ',\n'.join([event, target])
        self.short_data = (expect['id'] if expect else self.res['id'], expect['enabled'] if expect else self.res['enabled'], expect['name'] if expect else self.res['name'], expect['resource_type'] if expect else self.res['resource_type'], summary)

    def setUp(self):
        super(TestListNetworkLog, self).setUp()
        self.cmd = network_log.ListNetworkLog(self.app, self.namespace)
        self.short_header = ('ID', 'Enabled', 'Name', 'Type', 'Summary')
        self._setup_summary()
        self.neutronclient.list_network_logs = mock.Mock(return_value={'logs': [self.res]})
        self.mocked = self.neutronclient.list_network_logs

    def test_list_with_long_option(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with()
        self.assertEqual(list(self.headers), headers)
        self.assertEqual([self.data], list(data))

    def test_list_with_no_option(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with()
        self.assertEqual(list(self.short_header), headers)
        self.assertEqual([self.short_data], list(data))

    def test_list_with_target_and_resource(self):
        arglist = []
        verifylist = []
        target_id = 'aaaaaaaa-aaaa-aaaa-aaaaaaaaaaaaaaaaa'
        resource_id = 'bbbbbbbb-bbbb-bbbb-bbbbbbbbbbbbbbbbb'
        log = fakes.NetworkLog().create({'target_id': target_id, 'resource_id': resource_id})
        self.mocked.return_value = {'logs': [log]}
        logged = 'Logged: (security_group) %(res_id)s on (port) %(t_id)s' % {'res_id': resource_id, 't_id': target_id}
        expect_log = copy.deepcopy(log)
        expect_log.update({'resource': logged, 'event': 'Event: ALL'})
        self._setup_summary(expect=expect_log)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with()
        self.assertEqual(list(self.short_header), headers)
        self.assertEqual([self.short_data], list(data))

    def test_list_with_resource(self):
        arglist = []
        verifylist = []
        resource_id = 'bbbbbbbb-bbbb-bbbb-bbbbbbbbbbbbbbbbb'
        log = fakes.NetworkLog().create({'resource_id': resource_id})
        self.mocked.return_value = {'logs': [log]}
        logged = 'Logged: (security_group) %s' % resource_id
        expect_log = copy.deepcopy(log)
        expect_log.update({'resource': logged, 'event': 'Event: ALL'})
        self._setup_summary(expect=expect_log)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with()
        self.assertEqual(list(self.short_header), headers)
        self.assertEqual([self.short_data], list(data))

    def test_list_with_target(self):
        arglist = []
        verifylist = []
        target_id = 'aaaaaaaa-aaaa-aaaa-aaaaaaaaaaaaaaaaa'
        log = fakes.NetworkLog().create({'target_id': target_id})
        self.mocked.return_value = {'logs': [log]}
        logged = 'Logged: (port) %s' % target_id
        expect_log = copy.deepcopy(log)
        expect_log.update({'resource': logged, 'event': 'Event: ALL'})
        self._setup_summary(expect=expect_log)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        headers, data = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with()
        self.assertEqual(list(self.short_header), headers)
        self.assertEqual([self.short_data], list(data))