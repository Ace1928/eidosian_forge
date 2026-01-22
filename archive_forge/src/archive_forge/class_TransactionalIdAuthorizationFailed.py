import inspect
import sys
class TransactionalIdAuthorizationFailed(BrokerResponseError):
    errno = 53
    message = 'TRANSACTIONAL_ID_AUTHORIZATION_FAILED'
    description = 'Transactional Id authorization failed'