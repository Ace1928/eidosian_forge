from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RestResource(_messages.Message):
    """A RestResource object.

  Messages:
    MethodsValue: Methods on this resource.
    ResourcesValue: Sub-resources on this resource.

  Fields:
    methods: Methods on this resource.
    resources: Sub-resources on this resource.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MethodsValue(_messages.Message):
        """Methods on this resource.

    Messages:
      AdditionalProperty: An additional property for a MethodsValue object.

    Fields:
      additionalProperties: Description for any methods on this resource.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MethodsValue object.

      Fields:
        key: Name of the additional property.
        value: A RestMethod attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('RestMethod', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResourcesValue(_messages.Message):
        """Sub-resources on this resource.

    Messages:
      AdditionalProperty: An additional property for a ResourcesValue object.

    Fields:
      additionalProperties: Description for any sub-resources on this
        resource.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResourcesValue object.

      Fields:
        key: Name of the additional property.
        value: A RestResource attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('RestResource', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    methods = _messages.MessageField('MethodsValue', 1)
    resources = _messages.MessageField('ResourcesValue', 2)