from unittest import mock
from heat.common import exception
from heat.common import service_utils
from heat.engine import stack_lock
from heat.objects import stack as stack_object
from heat.objects import stack_lock as stack_lock_object
from heat.tests import common
from heat.tests import utils
def test_successful_acquire_new_lock(self):
    mock_create = self.patchobject(stack_lock_object.StackLock, 'create', return_value=None)
    slock = stack_lock.StackLock(self.context, self.stack_id, self.engine_id)
    slock.acquire()
    mock_create.assert_called_once_with(self.context, self.stack_id, self.engine_id)