from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsSearchNearestEntitiesRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsSearchNeares
  tEntitiesRequest object.

  Fields:
    featureView: Required. FeatureView resource format `projects/{project}/loc
      ations/{location}/featureOnlineStores/{featureOnlineStore}/featureViews/
      {featureView}`
    googleCloudAiplatformV1SearchNearestEntitiesRequest: A
      GoogleCloudAiplatformV1SearchNearestEntitiesRequest resource to be
      passed as the request body.
  """
    featureView = _messages.StringField(1, required=True)
    googleCloudAiplatformV1SearchNearestEntitiesRequest = _messages.MessageField('GoogleCloudAiplatformV1SearchNearestEntitiesRequest', 2)