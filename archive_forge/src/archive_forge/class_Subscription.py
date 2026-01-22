from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class Subscription(_messages.Message):
    """A subscription resource.

  Fields:
    ackDeadlineSeconds: This value is the maximum time after a subscriber
      receives a message before the subscriber should acknowledge the message.
      After message delivery but before the ack deadline expires and before
      the message is acknowledged, it is an outstanding message and will not
      be delivered again during that time (on a best-effort basis).  For pull
      subscriptions, this value is used as the initial value for the ack
      deadline. To override this value for a given message, call
      `ModifyAckDeadline` with the corresponding `ack_id` if using non-
      streaming pull or send the `ack_id` in a
      `StreamingModifyAckDeadlineRequest` if using streaming pull. The minimum
      custom deadline you can specify is 10 seconds. The maximum custom
      deadline you can specify is 600 seconds (10 minutes). If this parameter
      is 0, a default value of 10 seconds is used.  For push delivery, this
      value is also used to set the request timeout for the call to the push
      endpoint.  If the subscriber never acknowledges the message, the Pub/Sub
      system will eventually redeliver the message.
    name: The name of the subscription. It must have the format
      `"projects/{project}/subscriptions/{subscription}"`. `{subscription}`
      must start with a letter, and contain only letters (`[A-Za-z]`), numbers
      (`[0-9]`), dashes (`-`), underscores (`_`), periods (`.`), tildes (`~`),
      plus (`+`) or percent signs (`%`). It must be between 3 and 255
      characters in length, and it must not start with `"goog"`.
    pushConfig: If push delivery is used with this subscription, this field is
      used to configure it. An empty `pushConfig` signifies that the
      subscriber will pull and ack messages using API methods.
    topic: The name of the topic from which this subscription is receiving
      messages. Format is `projects/{project}/topics/{topic}`. The value of
      this field will be `_deleted-topic_` if the topic has been deleted.
  """
    ackDeadlineSeconds = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    name = _messages.StringField(2)
    pushConfig = _messages.MessageField('PushConfig', 3)
    topic = _messages.StringField(4)