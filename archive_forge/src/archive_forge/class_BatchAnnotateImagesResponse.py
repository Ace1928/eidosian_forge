from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchAnnotateImagesResponse(_messages.Message):
    """Response to a batch image annotation request.

  Fields:
    responses: Individual responses to image annotation requests within the
      batch.
  """
    responses = _messages.MessageField('AnnotateImageResponse', 1, repeated=True)