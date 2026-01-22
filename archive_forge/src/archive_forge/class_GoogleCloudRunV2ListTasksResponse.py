from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2ListTasksResponse(_messages.Message):
    """Response message containing a list of Tasks.

  Fields:
    nextPageToken: A token indicating there are more items than page_size. Use
      it in the next ListTasks request to continue.
    tasks: The resulting list of Tasks.
  """
    nextPageToken = _messages.StringField(1)
    tasks = _messages.MessageField('GoogleCloudRunV2Task', 2, repeated=True)