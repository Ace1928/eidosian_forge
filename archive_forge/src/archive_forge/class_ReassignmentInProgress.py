import inspect
import sys
class ReassignmentInProgress(BrokerResponseError):
    errno = 60
    message = 'REASSIGNMENT_IN_PROGRESS'
    description = 'A partition reassignment is in progress'