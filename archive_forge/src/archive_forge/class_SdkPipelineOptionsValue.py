from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class SdkPipelineOptionsValue(_messages.Message):
    """The Cloud Dataflow SDK pipeline options specified by the user. These
    options are passed through the service and are used to recreate the SDK
    pipeline options on the worker in a language agnostic and platform
    independent way.

    Messages:
      AdditionalProperty: An additional property for a SdkPipelineOptionsValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a SdkPipelineOptionsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)