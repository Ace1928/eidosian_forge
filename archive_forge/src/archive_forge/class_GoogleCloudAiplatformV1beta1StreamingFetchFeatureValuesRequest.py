from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1StreamingFetchFeatureValuesRequest(_messages.Message):
    """Request message for
  FeatureOnlineStoreService.StreamingFetchFeatureValues. For the entities
  requested, all features under the requested feature view will be returned.

  Enums:
    DataFormatValueValuesEnum: Specify response data format. If not set,
      KeyValue format will be used.

  Fields:
    dataFormat: Specify response data format. If not set, KeyValue format will
      be used.
    dataKeys: A GoogleCloudAiplatformV1beta1FeatureViewDataKey attribute.
  """

    class DataFormatValueValuesEnum(_messages.Enum):
        """Specify response data format. If not set, KeyValue format will be
    used.

    Values:
      FEATURE_VIEW_DATA_FORMAT_UNSPECIFIED: Not set. Will be treated as the
        KeyValue format.
      KEY_VALUE: Return response data in key-value format.
      PROTO_STRUCT: Return response data in proto Struct format.
    """
        FEATURE_VIEW_DATA_FORMAT_UNSPECIFIED = 0
        KEY_VALUE = 1
        PROTO_STRUCT = 2
    dataFormat = _messages.EnumField('DataFormatValueValuesEnum', 1)
    dataKeys = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureViewDataKey', 2, repeated=True)