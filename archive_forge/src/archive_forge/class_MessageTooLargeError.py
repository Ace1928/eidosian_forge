from __future__ import absolute_import
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.pubsub_v1.exceptions import TimeoutError
class MessageTooLargeError(ValueError):
    """Attempt to publish a message that would exceed the server max size limit."""