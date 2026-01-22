from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsDeleteRequest(_messages.Message):
    """A DatacatalogProjectsLocationsEntryGroupsDeleteRequest object.

  Fields:
    force: Optional. If true, deletes all entries in the entry group.
    name: Required. The name of the entry group to delete.
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)