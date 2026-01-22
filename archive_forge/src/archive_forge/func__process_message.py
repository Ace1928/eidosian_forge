from kombu import exceptions as kombu_exc
from taskflow import exceptions as excp
from taskflow import logging
from taskflow.utils import kombu_utils as ku
def _process_message(self, data, message, message_type):
    handler = self._type_handlers.get(message_type)
    if handler is None:
        message.reject_log_error(logger=LOG, errors=(kombu_exc.MessageStateError,))
        LOG.warning("Unexpected message type: '%s' in message '%s'", message_type, ku.DelayedPretty(message))
    else:
        if handler.validator is not None:
            try:
                handler.validator(data)
            except excp.InvalidFormat as e:
                message.reject_log_error(logger=LOG, errors=(kombu_exc.MessageStateError,))
                LOG.warning("Message '%s' (%s) was rejected due to it being in an invalid format: %s", ku.DelayedPretty(message), message_type, e)
                return
        message.ack_log_error(logger=LOG, errors=(kombu_exc.MessageStateError,))
        if message.acknowledged:
            LOG.debug("Message '%s' was acknowledged.", ku.DelayedPretty(message))
            handler.process_message(data, message)
        else:
            message.reject_log_error(logger=LOG, errors=(kombu_exc.MessageStateError,))