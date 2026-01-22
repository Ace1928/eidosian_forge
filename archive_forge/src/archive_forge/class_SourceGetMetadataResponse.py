from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceGetMetadataResponse(_messages.Message):
    """The result of a SourceGetMetadataOperation.

  Fields:
    metadata: The computed metadata.
  """
    metadata = _messages.MessageField('SourceMetadata', 1)