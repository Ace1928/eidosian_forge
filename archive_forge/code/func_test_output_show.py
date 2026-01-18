from unittest import mock
import testscenarios
from testscenarios import scenarios as scnrs
import testtools
from heatclient.v1 import stacks
def test_output_show(self):
    manager = mock.MagicMock()
    stack = mock_stack(manager, 'the_stack', 'abcd1234')
    stack.output_show('out123')
    manager.output_show.assert_called_once_with('the_stack/abcd1234', 'out123')