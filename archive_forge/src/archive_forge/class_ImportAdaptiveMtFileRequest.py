from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportAdaptiveMtFileRequest(_messages.Message):
    """The request for importing an AdaptiveMt file along with its sentences.

  Fields:
    fileInputSource: Inline file source.
    gcsInputSource: Google Cloud Storage file source.
  """
    fileInputSource = _messages.MessageField('FileInputSource', 1)
    gcsInputSource = _messages.MessageField('GcsInputSource', 2)