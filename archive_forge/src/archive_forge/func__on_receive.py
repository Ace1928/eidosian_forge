import functools
from oslo_utils import reflection
from oslo_utils import timeutils
from taskflow.engines.worker_based import dispatcher
from taskflow.engines.worker_based import protocol as pr
from taskflow.engines.worker_based import proxy
from taskflow import logging
from taskflow.types import failure as ft
from taskflow.types import notifier as nt
from taskflow.utils import kombu_utils as ku
from taskflow.utils import misc
def _on_receive(content, message):
    LOG.debug("Submitting message '%s' for execution in the future to '%s'", ku.DelayedPretty(message), func_name)
    watch = timeutils.StopWatch()
    watch.start()
    try:
        self._executor.submit(_on_run, watch, content, message)
    except RuntimeError:
        LOG.error("Unable to continue processing message '%s', submission to instance executor (with later execution by '%s') was unsuccessful", ku.DelayedPretty(message), func_name, exc_info=True)