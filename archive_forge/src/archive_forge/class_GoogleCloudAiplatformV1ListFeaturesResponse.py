from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListFeaturesResponse(_messages.Message):
    """Response message for FeaturestoreService.ListFeatures. Response message
  for FeatureRegistryService.ListFeatures.

  Fields:
    features: The Features matching the request.
    nextPageToken: A token, which can be sent as
      ListFeaturesRequest.page_token to retrieve the next page. If this field
      is omitted, there are no subsequent pages.
  """
    features = _messages.MessageField('GoogleCloudAiplatformV1Feature', 1, repeated=True)
    nextPageToken = _messages.StringField(2)