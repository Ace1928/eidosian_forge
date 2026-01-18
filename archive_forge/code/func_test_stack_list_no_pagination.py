from unittest import mock
import testscenarios
from testscenarios import scenarios as scnrs
import testtools
from heatclient.v1 import stacks
def test_stack_list_no_pagination(self):
    manager = self.mock_manager()
    results = list(manager.list())
    manager._list.assert_called_once_with('/stacks?', 'stacks')
    self.assertEqual(self.total, len(results))
    if self.total > 0:
        self.assertEqual('stack_1', results[0].stack_name)
        self.assertEqual('stack_%s' % self.total, results[-1].stack_name)