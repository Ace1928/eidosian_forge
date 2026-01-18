from unittest import mock
import testscenarios
from testscenarios import scenarios as scnrs
import testtools
from heatclient.v1 import stacks
def test_status_action(self):
    stack_status = '%s_%s' % (self.action, self.status)
    stack = mock_stack(None, 'stack_1', 'abcd1234')
    stack.stack_status = stack_status
    self.assertEqual(self.action, stack.action)
    self.assertEqual(self.status, stack.status)