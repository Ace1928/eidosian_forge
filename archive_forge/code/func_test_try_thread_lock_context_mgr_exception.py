from unittest import mock
from heat.common import exception
from heat.common import service_utils
from heat.engine import stack_lock
from heat.objects import stack as stack_object
from heat.objects import stack_lock as stack_lock_object
from heat.tests import common
from heat.tests import utils
def test_try_thread_lock_context_mgr_exception(self):
    stack_lock_object.StackLock.create = mock.Mock(return_value=None)
    stack_lock_object.StackLock.release = mock.Mock(return_value=None)
    slock = stack_lock.StackLock(self.context, self.stack_id, self.engine_id)

    def check_thread_lock():
        with slock.try_thread_lock():
            self.assertEqual(1, stack_lock_object.StackLock.create.call_count)
            raise self.TestThreadLockException
    self.assertRaises(self.TestThreadLockException, check_thread_lock)
    self.assertEqual(1, stack_lock_object.StackLock.release.call_count)