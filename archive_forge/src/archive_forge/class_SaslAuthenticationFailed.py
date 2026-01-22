import inspect
import sys
class SaslAuthenticationFailed(BrokerResponseError):
    errno = 58
    message = 'SASL_AUTHENTICATION_FAILED'
    description = 'SASL Authentication failed.'