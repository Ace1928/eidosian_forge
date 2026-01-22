import inspect
import sys
class OffsetOutOfRangeError(BrokerResponseError):
    errno = 1
    message = 'OFFSET_OUT_OF_RANGE'
    description = 'The requested offset is outside the range of offsets maintained by the server for the given topic/partition.'