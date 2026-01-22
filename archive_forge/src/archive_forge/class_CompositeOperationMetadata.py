from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CompositeOperationMetadata(_messages.Message):
    """Metadata for composite operations.

  Messages:
    OriginalRequestValue: Original request that triggered this operation.
    ResponseFieldMasksValue: Defines which part of the response a child
      operation will contribute. Each key of the map is the name of a child
      operation. Each value is a field mask that identifies what that child
      operation contributes to the response, for example, "quota_settings",
      "visiblity_settings", etc.

  Fields:
    childOperations: The child operations. The details of the asynchronous
      child operations are stored in a separate row and not in this metadata.
      Only the operation name is stored here.
    originalRequest: Original request that triggered this operation.
    persisted: Indicates whether the requested state change has been
      persisted. Once this field is set, it is guaranteed to propagate to all
      backends eventually, but it may not be visible immediately. Clients that
      are not concerned with waiting on propagation can stop polling the
      operation once the persisted field is set
    responseFieldMasks: Defines which part of the response a child operation
      will contribute. Each key of the map is the name of a child operation.
      Each value is a field mask that identifies what that child operation
      contributes to the response, for example, "quota_settings",
      "visiblity_settings", etc.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class OriginalRequestValue(_messages.Message):
        """Original request that triggered this operation.

    Messages:
      AdditionalProperty: An additional property for a OriginalRequestValue
        object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a OriginalRequestValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResponseFieldMasksValue(_messages.Message):
        """Defines which part of the response a child operation will contribute.
    Each key of the map is the name of a child operation. Each value is a
    field mask that identifies what that child operation contributes to the
    response, for example, "quota_settings", "visiblity_settings", etc.

    Messages:
      AdditionalProperty: An additional property for a ResponseFieldMasksValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        ResponseFieldMasksValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResponseFieldMasksValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    childOperations = _messages.MessageField('Operation', 1, repeated=True)
    originalRequest = _messages.MessageField('OriginalRequestValue', 2)
    persisted = _messages.BooleanField(3)
    responseFieldMasks = _messages.MessageField('ResponseFieldMasksValue', 4)