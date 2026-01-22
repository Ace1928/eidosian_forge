from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootControlDecodingRecords(_messages.Message):
    """A LearningGenaiRootControlDecodingRecords object.

  Fields:
    records: One ControlDecodingRecord record maps to one rewind.
  """
    records = _messages.MessageField('LearningGenaiRootControlDecodingRecord', 1, repeated=True)