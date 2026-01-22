from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SyncFeatureViewResponse(_messages.Message):
    """Respose message for FeatureOnlineStoreAdminService.SyncFeatureView.

  Fields:
    featureViewSync: Format: `projects/{project}/locations/{location}/featureO
      nlineStores/{feature_online_store}/featureViews/{feature_view}/featureVi
      ewSyncs/{feature_view_sync}`
  """
    featureViewSync = _messages.StringField(1)