import inspect
import sys
class NotEnoughReplicasError(BrokerResponseError):
    errno = 19
    message = 'NOT_ENOUGH_REPLICAS'
    description = 'Returned from a produce request when the number of in-sync replicas is lower than the configured minimum and requiredAcks is -1.'
    retriable = True