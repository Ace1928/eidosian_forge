from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudtasksProjectsLocationsQueuesTasksListRequest(_messages.Message):
    """A CloudtasksProjectsLocationsQueuesTasksListRequest object.

  Enums:
    ResponseViewValueValuesEnum: The response_view specifies which subset of
      the Task will be returned. By default response_view is BASIC; not all
      information is retrieved by default because some data, such as payloads,
      might be desirable to return only when needed because of its large size
      or because of the sensitivity of data that it contains. Authorization
      for FULL requires `cloudtasks.tasks.fullView` [Google
      IAM](https://cloud.google.com/iam/) permission on the Task resource.

  Fields:
    pageSize: Maximum page size. Fewer tasks than requested might be returned,
      even if more tasks exist; use next_page_token in the response to
      determine if more tasks exist. The maximum page size is 1000. If
      unspecified, the page size will be the maximum.
    pageToken: A token identifying the page of results to return. To request
      the first page results, page_token must be empty. To request the next
      page of results, page_token must be the value of next_page_token
      returned from the previous call to ListTasks method. The page token is
      valid for only 2 hours.
    parent: Required. The queue name. For example:
      `projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID`
    responseView: The response_view specifies which subset of the Task will be
      returned. By default response_view is BASIC; not all information is
      retrieved by default because some data, such as payloads, might be
      desirable to return only when needed because of its large size or
      because of the sensitivity of data that it contains. Authorization for
      FULL requires `cloudtasks.tasks.fullView` [Google
      IAM](https://cloud.google.com/iam/) permission on the Task resource.
  """

    class ResponseViewValueValuesEnum(_messages.Enum):
        """The response_view specifies which subset of the Task will be returned.
    By default response_view is BASIC; not all information is retrieved by
    default because some data, such as payloads, might be desirable to return
    only when needed because of its large size or because of the sensitivity
    of data that it contains. Authorization for FULL requires
    `cloudtasks.tasks.fullView` [Google IAM](https://cloud.google.com/iam/)
    permission on the Task resource.

    Values:
      VIEW_UNSPECIFIED: Unspecified. Defaults to BASIC.
      BASIC: The basic view omits fields which can be large or can contain
        sensitive data. This view does not include the body in
        AppEngineHttpRequest. Bodies are desirable to return only when needed,
        because they can be large and because of the sensitivity of the data
        that you choose to store in it.
      FULL: All information is returned. Authorization for FULL requires
        `cloudtasks.tasks.fullView` [Google
        IAM](https://cloud.google.com/iam/) permission on the Queue resource.
    """
        VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    responseView = _messages.EnumField('ResponseViewValueValuesEnum', 4)