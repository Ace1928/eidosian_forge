from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FeatureViewSync(_messages.Message):
    """FeatureViewSync is a representation of sync operation which copies data
  from data source to Feature View in Online Store.

  Fields:
    createTime: Output only. Time when this FeatureViewSync is created.
      Creation of a FeatureViewSync means that the job is pending / waiting
      for sufficient resources but may not have started the actual data
      transfer yet.
    finalStatus: Output only. Final status of the FeatureViewSync.
    name: Identifier. Name of the FeatureViewSync. Format: `projects/{project}
      /locations/{location}/featureOnlineStores/{feature_online_store}/feature
      Views/{feature_view}/featureViewSyncs/{feature_view_sync}`
    runTime: Output only. Time when this FeatureViewSync is finished.
    syncSummary: Output only. Summary of the sync job.
  """
    createTime = _messages.StringField(1)
    finalStatus = _messages.MessageField('GoogleRpcStatus', 2)
    name = _messages.StringField(3)
    runTime = _messages.MessageField('GoogleTypeInterval', 4)
    syncSummary = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureViewSyncSyncSummary', 5)