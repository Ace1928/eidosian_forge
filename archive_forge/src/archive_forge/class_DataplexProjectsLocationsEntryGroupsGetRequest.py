from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsEntryGroupsGetRequest(_messages.Message):
    """A DataplexProjectsLocationsEntryGroupsGetRequest object.

  Fields:
    name: Required. The resource name of the EntryGroup: projects/{project_num
      ber}/locations/{location_id}/entryGroups/{entry_group_id}.
  """
    name = _messages.StringField(1, required=True)