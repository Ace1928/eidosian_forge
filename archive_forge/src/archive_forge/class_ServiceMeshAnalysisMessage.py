from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceMeshAnalysisMessage(_messages.Message):
    """AnalysisMessage is a single message produced by an analyzer, and it used
  to communicate to the end user about the state of their Service Mesh
  configuration.

  Messages:
    ArgsValue: A UI can combine these args with a template (based on
      message_base.type) to produce an internationalized message.

  Fields:
    args: A UI can combine these args with a template (based on
      message_base.type) to produce an internationalized message.
    description: A human readable description of what the error means. It is
      suitable for non-internationalize display purposes.
    messageBase: Details common to all types of Istio and ServiceMesh analysis
      messages.
    resourcePaths: A list of strings specifying the resource identifiers that
      were the cause of message generation. A "path" here may be: *
      MEMBERSHIP_ID if the cause is a specific member cluster *
      MEMBERSHIP_ID/(NAMESPACE\\/)?RESOURCETYPE/NAME if the cause is a resource
      in a cluster
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ArgsValue(_messages.Message):
        """A UI can combine these args with a template (based on
    message_base.type) to produce an internationalized message.

    Messages:
      AdditionalProperty: An additional property for a ArgsValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ArgsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    args = _messages.MessageField('ArgsValue', 1)
    description = _messages.StringField(2)
    messageBase = _messages.MessageField('ServiceMeshAnalysisMessageBase', 3)
    resourcePaths = _messages.StringField(4, repeated=True)