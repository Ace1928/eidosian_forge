from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExplanationSpecOverride(_messages.Message):
    """The ExplanationSpec entries that can be overridden at online explanation
  time.

  Fields:
    examplesOverride: The example-based explanations parameter overrides.
    metadata: The metadata to be overridden. If not specified, no metadata is
      overridden.
    parameters: The parameters to be overridden. Note that the attribution
      method cannot be changed. If not specified, no parameter is overridden.
  """
    examplesOverride = _messages.MessageField('GoogleCloudAiplatformV1ExamplesOverride', 1)
    metadata = _messages.MessageField('GoogleCloudAiplatformV1ExplanationMetadataOverride', 2)
    parameters = _messages.MessageField('GoogleCloudAiplatformV1ExplanationParameters', 3)