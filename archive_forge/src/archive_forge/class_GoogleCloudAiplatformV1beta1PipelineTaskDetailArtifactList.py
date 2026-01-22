from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PipelineTaskDetailArtifactList(_messages.Message):
    """A list of artifact metadata.

  Fields:
    artifacts: Output only. A list of artifact metadata.
  """
    artifacts = _messages.MessageField('GoogleCloudAiplatformV1beta1Artifact', 1, repeated=True)