import functools
import threading
from oslo_utils import timeutils
from taskflow.engines.action_engine import executor
from taskflow.engines.worker_based import dispatcher
from taskflow.engines.worker_based import protocol as pr
from taskflow.engines.worker_based import proxy
from taskflow.engines.worker_based import types as wt
from taskflow import exceptions as exc
from taskflow import logging
from taskflow.task import EVENT_UPDATE_PROGRESS  # noqa
from taskflow.utils import kombu_utils as ku
from taskflow.utils import misc
from taskflow.utils import threading_utils as tu
def _publish_request(self, request, worker):
    """Publish request to a given topic."""
    LOG.debug("Submitting execution of '%s' to worker '%s' (expecting response identified by reply_to=%s and correlation_id=%s) - waited %0.3f seconds to get published", request, worker, self._uuid, request.uuid, timeutils.now() - request.created_on)
    try:
        self._proxy.publish(request, worker.topic, reply_to=self._uuid, correlation_id=request.uuid)
    except Exception:
        with misc.capture_failure() as failure:
            LOG.critical("Failed to submit '%s' (transitioning it to %s)", request, pr.FAILURE, exc_info=True)
            if request.transition_and_log_error(pr.FAILURE, logger=LOG):
                with self._ongoing_requests_lock:
                    del self._ongoing_requests[request.uuid]
                request.set_result(failure)