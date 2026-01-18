from unittest import mock
from oslo_db import exception
from heat.engine import sync_point
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_sync_with_time_throttle(self):
    ctx = utils.dummy_context()
    stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
    stack.converge_stack(stack.t, action=stack.CREATE)
    mock_sleep_time = self.sync_with_sleep(ctx, stack)
    self.assertTrue(mock_sleep_time.called)