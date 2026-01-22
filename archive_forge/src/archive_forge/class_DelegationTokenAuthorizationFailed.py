import inspect
import sys
class DelegationTokenAuthorizationFailed(BrokerResponseError):
    errno = 65
    message = 'DELEGATION_TOKEN_AUTHORIZATION_FAILED'
    description = 'Delegation Token authorization failed.'