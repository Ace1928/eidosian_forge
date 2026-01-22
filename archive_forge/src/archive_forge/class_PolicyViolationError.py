import inspect
import sys
class PolicyViolationError(BrokerResponseError):
    errno = 44
    message = 'POLICY_VIOLATION'
    description = 'Request parameters do not satisfy the configured policy.'