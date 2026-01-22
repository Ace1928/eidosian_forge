import inspect
import sys
class NotControllerError(BrokerResponseError):
    errno = 41
    message = 'NOT_CONTROLLER'
    description = 'This is not the correct controller for this cluster.'
    retriable = True