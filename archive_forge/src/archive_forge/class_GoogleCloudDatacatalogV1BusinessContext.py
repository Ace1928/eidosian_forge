from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1BusinessContext(_messages.Message):
    """Business Context of the entry.

  Fields:
    contacts: Contact people for the entry.
    entryOverview: Entry overview fields for rich text descriptions of
      entries.
  """
    contacts = _messages.MessageField('GoogleCloudDatacatalogV1Contacts', 1)
    entryOverview = _messages.MessageField('GoogleCloudDatacatalogV1EntryOverview', 2)