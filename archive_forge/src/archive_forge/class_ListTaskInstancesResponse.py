from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTaskInstancesResponse(_messages.Message):
    """Response to `ListTaskInstancesRequest`.

  Fields:
    nextPageToken: The page token used to query for the next page if one
      exists.
    taskInstances: The list of tasks returned.
  """
    nextPageToken = _messages.StringField(1)
    taskInstances = _messages.MessageField('TaskInstance', 2, repeated=True)