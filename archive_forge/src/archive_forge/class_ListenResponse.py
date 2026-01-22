from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListenResponse(_messages.Message):
    """The response for Firestore.Listen.

  Fields:
    documentChange: A Document has changed.
    documentDelete: A Document has been deleted.
    documentRemove: A Document has been removed from a target (because it is
      no longer relevant to that target).
    filter: A filter to apply to the set of documents previously returned for
      the given target. Returned when documents may have been removed from the
      given target, but the exact documents are unknown.
    targetChange: Targets have changed.
  """
    documentChange = _messages.MessageField('DocumentChange', 1)
    documentDelete = _messages.MessageField('DocumentDelete', 2)
    documentRemove = _messages.MessageField('DocumentRemove', 3)
    filter = _messages.MessageField('ExistenceFilter', 4)
    targetChange = _messages.MessageField('TargetChange', 5)