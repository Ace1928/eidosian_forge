import copy
import io
from unittest import mock
from osc_lib import exceptions as exc
from osc_lib import utils
import testscenarios
import yaml
from heatclient.common import template_format
from heatclient import exc as heat_exc
from heatclient.osc.v1 import stack
from heatclient.tests import inline_templates
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import events
from heatclient.v1 import resources
from heatclient.v1 import stacks
class TestStackHookPoll(TestStack):
    stack = stacks.Stack(None, {'id': '1234', 'stack_name': 'my_stack', 'creation_time': '2013-08-04T20:57:55Z', 'updated_time': '2013-08-04T20:57:55Z', 'stack_status': 'CREATE_IN_PROGRESS'})
    resource = resources.Resource(None, {'resource_name': 'resource1', 'links': [{'href': 'http://heat.example.com:8004/resource1', 'rel': 'self'}, {'href': 'http://192.168.27.100:8004/my_stack', 'rel': 'stack'}], 'logical_resource_id': 'random_group', 'creation_time': '2015-12-03T16:50:56', 'resource_status': 'INIT_COMPLETE', 'updated_time': '2015-12-03T16:50:56', 'required_by': [], 'resource_status_reason': '', 'physical_resource_id': '', 'resource_type': 'OS::Heat::ResourceGroup', 'id': '1111'})
    columns = ['ID', 'Resource Status Reason', 'Resource Status', 'Event Time']
    event0 = events.Event(manager=None, info={'resource_name': 'my_stack', 'event_time': '2015-12-02T16:50:56', 'logical_resource_id': 'my_stack', 'resource_status': 'CREATE_IN_PROGRESS', 'resource_status_reason': 'Stack CREATE started', 'id': '1234'})
    event1 = events.Event(manager=None, info={'resource_name': 'resource1', 'event_time': '2015-12-03T19:59:58', 'logical_resource_id': 'resource1', 'resource_status': 'INIT_COMPLETE', 'resource_status_reason': 'CREATE paused until Hook pre-create is cleared', 'id': '1111'})
    row1 = ('resource1', '1111', 'CREATE paused until Hook pre-create is cleared', 'INIT_COMPLETE', '2015-12-03T19:59:58')

    def setUp(self):
        super(TestStackHookPoll, self).setUp()
        self.cmd = stack.StackHookPoll(self.app, None)
        self.mock_client.stacks.get.return_value = self.stack
        self.mock_client.events.list.return_value = [self.event0, self.event1]
        self.mock_client.resources.list.return_value = [self.resource]

    def test_hook_poll(self):
        expected_columns = ['Resource Name'] + self.columns
        expected_rows = [self.row1]
        arglist = ['my_stack']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, rows = self.cmd.take_action(parsed_args)
        self.assertEqual(expected_rows, list(rows))
        self.assertEqual(expected_columns, columns)

    def test_hook_poll_nested(self):
        expected_columns = ['Resource Name'] + self.columns + ['Stack Name']
        expected_rows = [self.row1 + ('my_stack',)]
        arglist = ['my_stack', '--nested-depth=10']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, rows = self.cmd.take_action(parsed_args)
        self.assertEqual(expected_rows, list(rows))
        self.assertEqual(expected_columns, columns)

    def test_hook_poll_nested_invalid(self):
        arglist = ['my_stack', '--nested-depth=ugly']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)