from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentChange(_messages.Message):
    """A Document has changed. May be the result of multiple writes, including
  deletes, that ultimately resulted in a new value for the Document. Multiple
  DocumentChange messages may be returned for the same logical change, if
  multiple targets are affected.

  Fields:
    document: The new state of the Document. If `mask` is set, contains only
      fields that were updated or added.
    removedTargetIds: A set of target IDs for targets that no longer match
      this document.
    targetIds: A set of target IDs of targets that match this document.
  """
    document = _messages.MessageField('Document', 1)
    removedTargetIds = _messages.IntegerField(2, repeated=True, variant=_messages.Variant.INT32)
    targetIds = _messages.IntegerField(3, repeated=True, variant=_messages.Variant.INT32)