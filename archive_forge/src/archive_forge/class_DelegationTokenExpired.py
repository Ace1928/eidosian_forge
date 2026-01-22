import inspect
import sys
class DelegationTokenExpired(BrokerResponseError):
    errno = 66
    message = 'DELEGATION_TOKEN_EXPIRED'
    description = 'Delegation Token is expired.'