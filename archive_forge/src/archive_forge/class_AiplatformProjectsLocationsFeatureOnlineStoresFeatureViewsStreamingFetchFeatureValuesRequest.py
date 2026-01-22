from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsStreamingFetchFeatureValuesRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsStreamingFet
  chFeatureValuesRequest object.

  Fields:
    featureView: Required. FeatureView resource format `projects/{project}/loc
      ations/{location}/featureOnlineStores/{featureOnlineStore}/featureViews/
      {featureView}`
    googleCloudAiplatformV1beta1StreamingFetchFeatureValuesRequest: A
      GoogleCloudAiplatformV1beta1StreamingFetchFeatureValuesRequest resource
      to be passed as the request body.
  """
    featureView = _messages.StringField(1, required=True)
    googleCloudAiplatformV1beta1StreamingFetchFeatureValuesRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1StreamingFetchFeatureValuesRequest', 2)