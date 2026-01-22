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
def _process_notify(self, notify, message):
    """Process notify message and reply back."""
    try:
        reply_to = message.properties['reply_to']
    except KeyError:
        LOG.warning("The 'reply_to' message property is missing in received notify message '%s'", ku.DelayedPretty(message), exc_info=True)
    else:
        response = pr.Notify(topic=self._topic, tasks=list(self._endpoints.keys()))
        try:
            self._proxy.publish(response, routing_key=reply_to)
        except Exception:
            LOG.critical("Failed to send reply to '%s' with notify response '%s'", reply_to, response, exc_info=True)