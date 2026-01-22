import inspect
import sys
class InvalidSessionTimeoutError(BrokerResponseError):
    errno = 26
    message = 'INVALID_SESSION_TIMEOUT'
    description = 'Return in join group when the requested session timeout is outside of the allowed range on the broker'