import inspect
import sys
class DelegationTokenRequestNotAllowed(BrokerResponseError):
    errno = 64
    message = 'DELEGATION_TOKEN_REQUEST_NOT_ALLOWED'
    description = 'Delegation Token requests are not allowed on PLAINTEXT/1-way SSL channels and on delegation token authenticated channels.'