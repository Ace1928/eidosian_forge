import inspect
import sys
class InvalidProducerEpoch(BrokerResponseError):
    errno = 47
    message = 'INVALID_PRODUCER_EPOCH'
    description = "Producer attempted an operation with an old epoch. Either there is a newer producer with the same transactionalId, or the producer's transaction has been expired by the broker."