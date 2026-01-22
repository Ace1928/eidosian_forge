from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsEntryGroupsEntriesDeleteRequest(_messages.Message):
    """A DataplexProjectsLocationsEntryGroupsEntriesDeleteRequest object.

  Fields:
    name: Required. The resource name of the Entry: projects/{project}/locatio
      ns/{location}/entryGroups/{entry_group}/entries/{entry}.
  """
    name = _messages.StringField(1, required=True)