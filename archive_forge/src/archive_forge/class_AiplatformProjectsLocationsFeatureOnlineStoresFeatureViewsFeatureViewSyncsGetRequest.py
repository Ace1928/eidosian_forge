from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsFeatureViewSyncsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsFeatureViewS
  yncsGetRequest object.

  Fields:
    name: Required. The name of the FeatureViewSync resource. Format: `project
      s/{project}/locations/{location}/featureOnlineStores/{feature_online_sto
      re}/featureViews/{feature_view}/featureViewSyncs/{feature_view_sync}`
  """
    name = _messages.StringField(1, required=True)