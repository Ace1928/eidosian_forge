import copy
import time
from unittest import mock
import fixtures
from keystoneauth1 import exceptions as kc_exceptions
from oslo_log import log as logging
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os.keystone import heat_keystoneclient as hkc
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.objects import snapshot as snapshot_object
from heat.objects import stack as stack_object
from heat.objects import user_creds as ucreds_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_stack_delete_timeout(self):
    self.stack = stack.Stack(self.ctx, 'delete_test', self.tmpl)
    stack_id = self.stack.store()
    db_s = stack_object.Stack.get_by_id(self.ctx, stack_id)
    self.assertIsNotNone(db_s)

    def dummy_task():
        while True:
            yield
    start_time = time.time()
    mock_tg = self.patchobject(scheduler.DependencyTaskGroup, '__call__', return_value=dummy_task())
    mock_wallclock = self.patchobject(timeutils, 'wallclock')
    mock_wallclock.side_effect = [start_time, start_time + 1, start_time + self.stack.timeout_secs() + 1]
    self.stack.delete()
    self.assertEqual((stack.Stack.DELETE, stack.Stack.FAILED), self.stack.state)
    self.assertEqual('Delete timed out', self.stack.status_reason)
    mock_tg.assert_called_once_with()
    mock_wallclock.assert_called_with()
    self.assertEqual(3, mock_wallclock.call_count)