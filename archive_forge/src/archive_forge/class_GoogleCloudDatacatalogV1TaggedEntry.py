from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1TaggedEntry(_messages.Message):
    """Wrapper containing Entry and information about Tags that should and
  should not be attached to it.

  Fields:
    absentTags: Optional. Tags that should be deleted from the Data Catalog.
      Caller should populate template name and column only.
    presentTags: Optional. Tags that should be ingested into the Data Catalog.
      Caller should populate template name, column and fields.
    v1Entry: Non-encrypted Data Catalog v1 Entry.
  """
    absentTags = _messages.MessageField('GoogleCloudDatacatalogV1Tag', 1, repeated=True)
    presentTags = _messages.MessageField('GoogleCloudDatacatalogV1Tag', 2, repeated=True)
    v1Entry = _messages.MessageField('GoogleCloudDatacatalogV1Entry', 3)