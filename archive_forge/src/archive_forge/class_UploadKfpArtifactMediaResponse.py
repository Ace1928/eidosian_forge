from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UploadKfpArtifactMediaResponse(_messages.Message):
    """The response to upload an artifact.

  Fields:
    operation: Operation that will be returned to the user.
  """
    operation = _messages.MessageField('Operation', 1)