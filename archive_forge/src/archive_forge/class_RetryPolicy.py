from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RetryPolicy(_messages.Message):
    """A policy that specifies how Pub/Sub retries message delivery. Retry
  delay will be exponential based on provided minimum and maximum backoffs.
  https://en.wikipedia.org/wiki/Exponential_backoff. RetryPolicy will be
  triggered on NACKs or acknowledgement deadline exceeded events for a given
  message. Retry Policy is implemented on a best effort basis. At times, the
  delay between consecutive deliveries may not match the configuration. That
  is, delay can be more or less than configured backoff.

  Fields:
    maximumBackoff: Optional. The maximum delay between consecutive deliveries
      of a given message. Value should be between 0 and 600 seconds. Defaults
      to 600 seconds.
    minimumBackoff: Optional. The minimum delay between consecutive deliveries
      of a given message. Value should be between 0 and 600 seconds. Defaults
      to 10 seconds.
  """
    maximumBackoff = _messages.StringField(1)
    minimumBackoff = _messages.StringField(2)