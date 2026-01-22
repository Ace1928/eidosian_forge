import inspect
import sys
class TopicAuthorizationFailedError(BrokerResponseError):
    errno = 29
    message = 'TOPIC_AUTHORIZATION_FAILED'
    description = 'Returned by the broker when the client is not authorized to access the requested topic.'