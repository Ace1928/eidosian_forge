import inspect
import sys
class InvalidReplicationAssignmentError(BrokerResponseError):
    errno = 39
    message = 'INVALID_REPLICATION_ASSIGNMENT'
    description = 'Replication assignment is invalid.'