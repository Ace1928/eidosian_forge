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
def _delayed_process(self, func):
    """Runs the function using the instances executor (eventually).

        This adds a *nice* benefit on showing how long it took for the
        function to finally be executed from when the message was received
        to when it was finally ran (which can be a nice thing to know
        to determine bottle-necks...).
        """
    func_name = reflection.get_callable_name(func)

    def _on_run(watch, content, message):
        LOG.trace("It took %s seconds to get around to running function/method '%s' with message '%s'", watch.elapsed(), func_name, ku.DelayedPretty(message))
        return func(content, message)

    def _on_receive(content, message):
        LOG.debug("Submitting message '%s' for execution in the future to '%s'", ku.DelayedPretty(message), func_name)
        watch = timeutils.StopWatch()
        watch.start()
        try:
            self._executor.submit(_on_run, watch, content, message)
        except RuntimeError:
            LOG.error("Unable to continue processing message '%s', submission to instance executor (with later execution by '%s') was unsuccessful", ku.DelayedPretty(message), func_name, exc_info=True)
    return _on_receive