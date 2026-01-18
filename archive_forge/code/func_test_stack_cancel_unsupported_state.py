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
def test_stack_cancel_unsupported_state(self):
    self.stack.stack_status = 'CREATE_COMPLETE'
    self.mock_client.stacks.get.return_value = self.stack
    error = self.assertRaises(exc.CommandError, self._test_stack_action, 2)
    self.assertEqual("Stack my_stack with status 'create_complete' not in cancelable state", str(error))