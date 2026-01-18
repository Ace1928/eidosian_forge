import contextlib
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common import exception
from heat.common import service_utils
from heat.objects import stack as stack_object
from heat.objects import stack_lock as stack_lock_object
Similar to thread_lock, but acquire the lock using try_acquire.

        Only release it upon any exception after a successful acquisition.
        