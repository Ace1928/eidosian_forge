from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsExecutionsListRequest(_messages.Message):
    """A NotebooksProjectsLocationsExecutionsListRequest object.

  Fields:
    filter: Filter applied to resulting executions. Currently only supports
      filtering executions by a specified `schedule_id`. Format:
      `schedule_id=`
    orderBy: Sort by field.
    pageSize: Maximum return size of the list call.
    pageToken: A previous returned page token that can be used to continue
      listing from the last result.
    parent: Required. Format:
      `parent=projects/{project_id}/locations/{location}`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)