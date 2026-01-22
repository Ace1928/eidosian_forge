import inspect
import sys
class UnsupportedSaslMechanismError(BrokerResponseError):
    errno = 33
    message = 'UNSUPPORTED_SASL_MECHANISM'
    description = 'The broker does not support the requested SASL mechanism.'