from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1CitationMetadata(_messages.Message):
    """A collection of source attributions for a piece of content.

  Fields:
    citations: Output only. List of citations.
  """
    citations = _messages.MessageField('GoogleCloudAiplatformV1beta1Citation', 1, repeated=True)