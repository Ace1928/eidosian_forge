import inspect
import sys
class NotEnoughReplicasAfterAppendError(BrokerResponseError):
    errno = 20
    message = 'NOT_ENOUGH_REPLICAS_AFTER_APPEND'
    description = 'Returned from a produce request when the message was written to the log, but with fewer in-sync replicas than required.'
    retriable = True