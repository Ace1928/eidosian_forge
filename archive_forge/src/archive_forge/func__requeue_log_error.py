from kombu import exceptions as kombu_exc
from taskflow import exceptions as excp
from taskflow import logging
from taskflow.utils import kombu_utils as ku
def _requeue_log_error(self, message, errors):
    try:
        message.requeue()
    except errors as exc:
        LOG.critical("Couldn't requeue %r, reason:%r", message.delivery_tag, exc, exc_info=True)
    else:
        LOG.debug("Message '%s' was requeued.", ku.DelayedPretty(message))