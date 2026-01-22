from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FetchFeatureValuesResponse(_messages.Message):
    """Response message for FeatureOnlineStoreService.FetchFeatureValues

  Messages:
    ProtoStructValue: Feature values in proto Struct format.

  Fields:
    dataKey: The data key associated with this response. Will only be
      populated for FeatureOnlineStoreService.StreamingFetchFeatureValues
      RPCs.
    keyValues: Feature values in KeyValue format.
    protoStruct: Feature values in proto Struct format.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ProtoStructValue(_messages.Message):
        """Feature values in proto Struct format.

    Messages:
      AdditionalProperty: An additional property for a ProtoStructValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ProtoStructValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    dataKey = _messages.MessageField('GoogleCloudAiplatformV1FeatureViewDataKey', 1)
    keyValues = _messages.MessageField('GoogleCloudAiplatformV1FetchFeatureValuesResponseFeatureNameValuePairList', 2)
    protoStruct = _messages.MessageField('ProtoStructValue', 3)