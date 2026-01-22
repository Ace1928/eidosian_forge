from __future__ import absolute_import
from google.api_core.exceptions import GoogleAPICallError
from google.cloud.pubsub_v1.exceptions import TimeoutError
class FlowControlLimitError(Exception):
    """An action resulted in exceeding the flow control limits."""