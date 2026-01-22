from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1PurgeArtifactsResponse(_messages.Message):
    """Response message for MetadataService.PurgeArtifacts.

  Fields:
    purgeCount: The number of Artifacts that this request deleted (or, if
      `force` is false, the number of Artifacts that will be deleted). This
      can be an estimate.
    purgeSample: A sample of the Artifact names that will be deleted. Only
      populated if `force` is set to false. The maximum number of samples is
      100 (it is possible to return fewer).
  """
    purgeCount = _messages.IntegerField(1)
    purgeSample = _messages.StringField(2, repeated=True)