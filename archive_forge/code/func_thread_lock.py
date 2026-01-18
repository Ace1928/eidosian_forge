import contextlib
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common import exception
from heat.common import service_utils
from heat.objects import stack as stack_object
from heat.objects import stack_lock as stack_lock_object
@contextlib.contextmanager
def thread_lock(self, retry=True):
    """Acquire a lock and release it only if there is an exception.

        The release method still needs to be scheduled to be run at the
        end of the thread using the Thread.link method.

        :param retry: When True, retry if lock was released while stealing.
        :type retry: boolean
        """
    try:
        self.acquire(retry)
        yield
    except exception.ActionInProgress:
        raise
    except:
        with excutils.save_and_reraise_exception():
            self.release()