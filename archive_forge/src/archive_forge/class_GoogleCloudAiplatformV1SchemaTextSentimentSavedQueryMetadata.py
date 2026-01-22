from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTextSentimentSavedQueryMetadata(_messages.Message):
    """The metadata of SavedQuery contains TextSentiment Annotations.

  Fields:
    sentimentMax: The maximum sentiment of sentiment Anntoation in this
      SavedQuery.
  """
    sentimentMax = _messages.IntegerField(1, variant=_messages.Variant.INT32)