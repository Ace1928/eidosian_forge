import re
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import units
import glance.async_
from glance.common import exception
from glance.i18n import _, _LE, _LW
@memoize(lock_name)
def _get_thread_pool():
    threadpool_cls = glance.async_.get_threadpool_model()
    LOG.debug('Initializing named threadpool %r', lock_name)
    return threadpool_cls(size)