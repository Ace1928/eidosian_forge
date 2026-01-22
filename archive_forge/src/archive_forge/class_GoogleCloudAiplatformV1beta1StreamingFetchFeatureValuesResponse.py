from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1StreamingFetchFeatureValuesResponse(_messages.Message):
    """Response message for
  FeatureOnlineStoreService.StreamingFetchFeatureValues.

  Fields:
    data: A GoogleCloudAiplatformV1beta1FetchFeatureValuesResponse attribute.
    dataKeysWithError: A GoogleCloudAiplatformV1beta1FeatureViewDataKey
      attribute.
    status: Response status. If OK, then
      StreamingFetchFeatureValuesResponse.data will be populated. Otherwise
      StreamingFetchFeatureValuesResponse.data_keys_with_error will be
      populated with the appropriate data keys. The error only applies to the
      listed data keys - the stream will remain open for further
      FeatureOnlineStoreService.StreamingFetchFeatureValuesRequest requests.
  """
    data = _messages.MessageField('GoogleCloudAiplatformV1beta1FetchFeatureValuesResponse', 1, repeated=True)
    dataKeysWithError = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureViewDataKey', 2, repeated=True)
    status = _messages.MessageField('GoogleRpcStatus', 3)