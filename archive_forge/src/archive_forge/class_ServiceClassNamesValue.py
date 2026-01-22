from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ServiceClassNamesValue(_messages.Message):
    """Configure the Java class name to use instead of the service's for its
    corresponding generated GAPIC client. Keys are fully-qualified service
    names as they appear in the protobuf (including the full the
    language_settings.java.interface_names" field in gapic.yaml. API teams
    should otherwise use the service name as it appears in the protobuf.
    Example of a YAML configuration:: publishing: java_settings:
    service_class_names: - google.pubsub.v1.Publisher: TopicAdmin -
    google.pubsub.v1.Subscriber: SubscriptionAdmin

    Messages:
      AdditionalProperty: An additional property for a ServiceClassNamesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        ServiceClassNamesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ServiceClassNamesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)