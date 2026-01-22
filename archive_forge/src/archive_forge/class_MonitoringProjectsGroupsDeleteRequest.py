from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsGroupsDeleteRequest(_messages.Message):
    """A MonitoringProjectsGroupsDeleteRequest object.

  Fields:
    name: Required. The group to delete. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID]
    recursive: If this field is true, then the request means to delete a group
      with all its descendants. Otherwise, the request means to delete a group
      only when it has no descendants. The default value is false.
  """
    name = _messages.StringField(1, required=True)
    recursive = _messages.BooleanField(2)