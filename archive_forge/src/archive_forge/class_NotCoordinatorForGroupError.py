import inspect
import sys
class NotCoordinatorForGroupError(BrokerResponseError):
    errno = 16
    message = 'NOT_COORDINATOR'
    description = 'The broker returns this error code if it receives an offset fetch or commit request for a group that it is not a coordinator for.'
    retriable = True