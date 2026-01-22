import inspect
import sys
class UnknownProducerId(BrokerResponseError):
    errno = 59
    message = 'UNKNOWN_PRODUCER_ID'
    description = "This exception is raised by the broker if it could not locate the producer metadata associated with the producerId in question. This could happen if, for instance, the producer's records were deleted because their retention time had elapsed. Once the last records of the producerId are removed, the producer's metadata is removed from the broker, and future appends by the producer will return this exception."