from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsGroupsGetRequest(_messages.Message):
    """A MonitoringProjectsGroupsGetRequest object.

  Fields:
    name: Required. The group to retrieve. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/groups/[GROUP_ID]
  """
    name = _messages.StringField(1, required=True)