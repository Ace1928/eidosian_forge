from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportAdaptiveMtFileResponse(_messages.Message):
    """The response for importing an AdaptiveMtFile

  Fields:
    adaptiveMtFile: Output only. The Adaptive MT file that was imported.
  """
    adaptiveMtFile = _messages.MessageField('AdaptiveMtFile', 1)