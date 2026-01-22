import inspect
import sys
class FetchSessionIdNotFound(BrokerResponseError):
    errno = 70
    message = 'FETCH_SESSION_ID_NOT_FOUND'
    description = 'The fetch session ID was not found'