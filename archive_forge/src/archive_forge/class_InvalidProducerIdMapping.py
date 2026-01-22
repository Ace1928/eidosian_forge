import inspect
import sys
class InvalidProducerIdMapping(BrokerResponseError):
    errno = 49
    message = 'INVALID_PRODUCER_ID_MAPPING'
    description = 'The producer attempted to use a producer id which is not currently assigned to its transactional id'