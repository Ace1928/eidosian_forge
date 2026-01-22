from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentProvenanceParent(_messages.Message):
    """The parent element the current element is based on. Used for
  referencing/aligning, removal and replacement operations.

  Fields:
    id: The id of the parent provenance.
    index: The index of the parent item in the corresponding item list (eg.
      list of entities, properties within entities, etc.) in the parent
      revision.
    revision: The index of the index into current revision's parent_ids list.
  """
    id = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    index = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    revision = _messages.IntegerField(3, variant=_messages.Variant.INT32)