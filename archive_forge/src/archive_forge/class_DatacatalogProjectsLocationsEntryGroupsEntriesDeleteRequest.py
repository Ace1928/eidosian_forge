from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsEntriesDeleteRequest(_messages.Message):
    """A DatacatalogProjectsLocationsEntryGroupsEntriesDeleteRequest object.

  Fields:
    name: Required. The name of the entry to delete.
  """
    name = _messages.StringField(1, required=True)