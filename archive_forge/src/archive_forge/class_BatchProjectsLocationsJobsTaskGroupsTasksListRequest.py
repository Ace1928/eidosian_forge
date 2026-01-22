from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchProjectsLocationsJobsTaskGroupsTasksListRequest(_messages.Message):
    """A BatchProjectsLocationsJobsTaskGroupsTasksListRequest object.

  Fields:
    filter: Task filter, null filter matches all Tasks. Filter string should
      be of the format State=TaskStatus.State e.g. State=RUNNING
    orderBy: Not implemented.
    pageSize: Page size.
    pageToken: Page token.
    parent: Required. Name of a TaskGroup from which Tasks are being
      requested. Pattern: "projects/{project}/locations/{location}/jobs/{job}/
      taskGroups/{task_group}"
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)