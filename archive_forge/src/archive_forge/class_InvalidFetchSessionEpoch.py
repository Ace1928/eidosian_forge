import inspect
import sys
class InvalidFetchSessionEpoch(BrokerResponseError):
    errno = 71
    message = 'INVALID_FETCH_SESSION_EPOCH'
    description = 'The fetch session epoch is invalid'