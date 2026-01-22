from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1ProcessResponse(_messages.Message):
    """Response message for the ProcessDocument method.

  Fields:
    document: The document payload, will populate fields based on the
      processor's behavior.
    humanReviewStatus: The status of human review on the processed document.
  """
    document = _messages.MessageField('GoogleCloudDocumentaiV1Document', 1)
    humanReviewStatus = _messages.MessageField('GoogleCloudDocumentaiV1HumanReviewStatus', 2)