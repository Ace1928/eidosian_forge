from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TimeTarget(_messages.Message):
    """A target publish or event time. Can be used for seeking to or retrieving
  the corresponding cursor.

  Fields:
    eventTime: Request the cursor of the first message with event time greater
      than or equal to `event_time`. If messages are missing an event time,
      the publish time is used as a fallback. As event times are user
      supplied, subsequent messages may have event times less than
      `event_time` and should be filtered by the client, if necessary.
    publishTime: Request the cursor of the first message with publish time
      greater than or equal to `publish_time`. All messages thereafter are
      guaranteed to have publish times >= `publish_time`.
  """
    eventTime = _messages.StringField(1)
    publishTime = _messages.StringField(2)