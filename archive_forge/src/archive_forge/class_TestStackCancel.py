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
class TestStackCancel(_TestStackCheckBase, TestStack):
    stack_update_in_progress = stacks.Stack(None, {'id': '1234', 'stack_name': 'my_stack', 'creation_time': '2013-08-04T20:57:55Z', 'updated_time': '2013-08-04T20:57:55Z', 'stack_status': 'UPDATE_IN_PROGRESS'})

    def setUp(self):
        super(TestStackCancel, self).setUp()
        self._setUp(stack.CancelStack(self.app, None), self.mock_client.actions.cancel_update, 'ROLLBACK')
        self.mock_client.stacks.get.return_value = self.stack_update_in_progress

    def test_stack_cancel(self):
        self._test_stack_action(2)

    def _test_stack_cancel_no_rollback(self, call_count):
        self.action = self.mock_client.actions.cancel_without_rollback
        arglist = ['my_stack', '--no-rollback']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, rows = self.cmd.take_action(parsed_args)
        self.action.assert_called_once_with('my_stack')
        self.mock_client.stacks.get.assert_called_with('my_stack')
        self.assertEqual(call_count, self.mock_client.stacks.get.call_count)
        self.assertEqual(self.columns, columns)
        self.assertEqual(1, len(rows))

    def test_stack_cancel_no_rollback(self):
        self._test_stack_cancel_no_rollback(2)

    def test_stack_cancel_multi(self):
        self._test_stack_action_multi(4)

    def test_stack_cancel_wait(self):
        self._test_stack_action_wait()

    def test_stack_cancel_wait_error(self):
        self._test_stack_action_wait_error()

    def test_stack_cancel_exception(self):
        self._test_stack_action_exception()

    def test_stack_cancel_unsupported_state(self):
        self.stack.stack_status = 'CREATE_COMPLETE'
        self.mock_client.stacks.get.return_value = self.stack
        error = self.assertRaises(exc.CommandError, self._test_stack_action, 2)
        self.assertEqual("Stack my_stack with status 'create_complete' not in cancelable state", str(error))

    def test_stack_cancel_create_in_progress(self):
        self.stack.stack_status = 'CREATE_IN_PROGRESS'
        self.mock_client.stacks.get.return_value = self.stack
        error = self.assertRaises(exc.CommandError, self._test_stack_action, 2)
        self.assertEqual("Stack my_stack with status 'create_in_progress' not in cancelable state", str(error))
        self._test_stack_cancel_no_rollback(3)