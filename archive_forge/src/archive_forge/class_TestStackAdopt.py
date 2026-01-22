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
class TestStackAdopt(TestStack):
    adopt_file = 'heatclient/tests/test_templates/adopt.json'
    adopt_with_files = 'heatclient/tests/test_templates/adopt_with_file.json'
    with open(adopt_file, 'r') as f:
        adopt_data = f.read()
    with open(adopt_with_files, 'r') as f:
        adopt_with_files_data = f.read()
    defaults = {'stack_name': 'my_stack', 'disable_rollback': True, 'adopt_stack_data': adopt_data, 'parameters': {}, 'files': {}, 'environment': {}, 'timeout': None}
    child_stack_yaml = '{"heat_template_version": "2015-10-15"}'
    expected_with_files = {'stack_name': 'my_stack', 'disable_rollback': True, 'adopt_stack_data': adopt_with_files_data, 'parameters': {}, 'files': {'file://empty.yaml': child_stack_yaml}, 'environment': {}, 'timeout': None}

    def setUp(self):
        super(TestStackAdopt, self).setUp()
        self.cmd = stack.AdoptStack(self.app, None)
        self.stack_client.create.return_value = {'stack': {'id': '1234'}}

    def test_stack_adopt_defaults(self):
        arglist = ['my_stack', '--adopt-file', self.adopt_file]
        cols = ['id', 'stack_name', 'description', 'creation_time', 'updated_time', 'stack_status', 'stack_status_reason']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.stack_client.create.assert_called_with(**self.defaults)
        self.assertEqual(cols, columns)

    def test_stack_adopt_enable_rollback(self):
        arglist = ['my_stack', '--adopt-file', self.adopt_file, '--enable-rollback']
        kwargs = copy.deepcopy(self.defaults)
        kwargs['disable_rollback'] = False
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.stack_client.create.assert_called_with(**kwargs)

    @mock.patch('heatclient.common.event_utils.poll_for_events', return_value=('ADOPT_COMPLETE', 'Stack my_stack ADOPT_COMPLETE'))
    def test_stack_adopt_wait(self, mock_poll):
        arglist = ['my_stack', '--adopt-file', self.adopt_file, '--wait']
        self.stack_client.get.return_value = stacks.Stack(None, {'stack_status': 'ADOPT_COMPLETE'})
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.stack_client.create.assert_called_with(**self.defaults)
        self.stack_client.get.assert_called_with(**{'stack_id': '1234', 'resolve_outputs': False})

    @mock.patch('heatclient.common.event_utils.poll_for_events', return_value=('ADOPT_FAILED', 'Stack my_stack ADOPT_FAILED'))
    def test_stack_adopt_wait_fail(self, mock_poll):
        arglist = ['my_stack', '--adopt-file', self.adopt_file, '--wait']
        self.stack_client.get.return_value = stacks.Stack(None, {'stack_status': 'ADOPT_FAILED'})
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)

    def test_stack_adopt_with_adopt_files(self):
        arglist = ['my_stack', '--adopt-file', self.adopt_with_files]
        cols = ['id', 'stack_name', 'description', 'creation_time', 'updated_time', 'stack_status', 'stack_status_reason']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.stack_client.create.assert_called_with(**self.expected_with_files)
        self.assertEqual(cols, columns)