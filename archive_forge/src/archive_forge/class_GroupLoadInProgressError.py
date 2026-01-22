import inspect
import sys
class GroupLoadInProgressError(BrokerResponseError):
    errno = 14
    message = 'COORDINATOR_LOAD_IN_PROGRESS'
    description = 'The broker returns this error code for an offset fetch request if it is still loading offsets (after a leader change for that offsets topic partition), or in response to group membership requests (such as heartbeats) when group metadata is being loaded by the coordinator.'
    retriable = True