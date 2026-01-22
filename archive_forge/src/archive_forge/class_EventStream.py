from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventStream(_messages.Message):
    """Specifies the Event-driven transfer options. Event-driven transfers
  listen to an event stream to transfer updated files.

  Fields:
    eventStreamExpirationTime: Specifies the data and time at which Storage
      Transfer Service stops listening for events from this stream. After this
      time, any transfers in progress will complete, but no new transfers are
      initiated.
    eventStreamStartTime: Specifies the date and time that Storage Transfer
      Service starts listening for events from this stream. If no start time
      is specified or start time is in the past, Storage Transfer Service
      starts listening immediately.
    name: Required. Specifies a unique name of the resource such as AWS SQS
      ARN in the form 'arn:aws:sqs:region:account_id:queue_name', or Pub/Sub
      subscription resource name in the form
      'projects/{project}/subscriptions/{sub}'.
  """
    eventStreamExpirationTime = _messages.StringField(1)
    eventStreamStartTime = _messages.StringField(2)
    name = _messages.StringField(3)