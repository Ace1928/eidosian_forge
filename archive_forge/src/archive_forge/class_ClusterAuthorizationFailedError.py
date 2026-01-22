import inspect
import sys
class ClusterAuthorizationFailedError(BrokerResponseError):
    errno = 31
    message = 'CLUSTER_AUTHORIZATION_FAILED'
    description = 'Returned by the broker when the client is not authorized to use an inter-broker or administrative API.'