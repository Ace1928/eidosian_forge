from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentRemove(_messages.Message):
    """A Document has been removed from the view of the targets. Sent if the
  document is no longer relevant to a target and is out of view. Can be sent
  instead of a DocumentDelete or a DocumentChange if the server can not send
  the new value of the document. Multiple DocumentRemove messages may be
  returned for the same logical write or delete, if multiple targets are
  affected.

  Fields:
    document: The resource name of the Document that has gone out of view.
    readTime: The read timestamp at which the remove was observed. Greater or
      equal to the `commit_time` of the change/delete/remove.
    removedTargetIds: A set of target IDs for targets that previously matched
      this document.
  """
    document = _messages.StringField(1)
    readTime = _messages.StringField(2)
    removedTargetIds = _messages.IntegerField(3, repeated=True, variant=_messages.Variant.INT32)