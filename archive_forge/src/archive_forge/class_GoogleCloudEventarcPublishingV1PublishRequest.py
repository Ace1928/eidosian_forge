from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudEventarcPublishingV1PublishRequest(_messages.Message):
    """The request message for the Publish method.

  Fields:
    avroMessage: The Avro format of the CloudEvent being published.
      Specification can be found here: https://github.com/cloudevents/spec/blo
      b/v1.0.2/cloudevents/formats/avro-format.md
    jsonMessage: The JSON format of the CloudEvent being published.
      Specification can be found here: https://github.com/cloudevents/spec/blo
      b/v1.0.2/cloudevents/formats/json-format.md
    protoMessage: The Protobuf format of the CloudEvent being published.
      Specification can be found here: https://github.com/cloudevents/spec/blo
      b/v1.0.2/cloudevents/formats/protobuf-format.md
  """
    avroMessage = _messages.BytesField(1)
    jsonMessage = _messages.StringField(2)
    protoMessage = _messages.MessageField('IoCloudeventsV1CloudEvent', 3)