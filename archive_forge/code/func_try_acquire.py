import contextlib
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common import exception
from heat.common import service_utils
from heat.objects import stack as stack_object
from heat.objects import stack_lock as stack_lock_object
def try_acquire(self):
    """Try to acquire a stack lock.

        Don't raise an ActionInProgress exception or try to steal lock.
        """
    return stack_lock_object.StackLock.create(self.context, self.stack_id, self.engine_id)