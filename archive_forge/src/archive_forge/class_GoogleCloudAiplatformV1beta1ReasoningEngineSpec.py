from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ReasoningEngineSpec(_messages.Message):
    """ReasoningEngine configurations

  Messages:
    ClassMethodsValueListEntry: A ClassMethodsValueListEntry object.

  Fields:
    classMethods: Optional. Declarations for object class methods.
    packageSpec: Required. User provided package spec of the ReasoningEngine.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ClassMethodsValueListEntry(_messages.Message):
        """A ClassMethodsValueListEntry object.

    Messages:
      AdditionalProperty: An additional property for a
        ClassMethodsValueListEntry object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ClassMethodsValueListEntry object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    classMethods = _messages.MessageField('ClassMethodsValueListEntry', 1, repeated=True)
    packageSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1ReasoningEngineSpecPackageSpec', 2)