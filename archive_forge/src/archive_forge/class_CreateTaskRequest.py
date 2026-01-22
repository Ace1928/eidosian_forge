from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateTaskRequest(_messages.Message):
    """Request message for CreateTask.

  Enums:
    ResponseViewValueValuesEnum: The response_view specifies which subset of
      the Task will be returned. By default response_view is BASIC; not all
      information is retrieved by default because some data, such as payloads,
      might be desirable to return only when needed because of its large size
      or because of the sensitivity of data that it contains. Authorization
      for FULL requires `cloudtasks.tasks.fullView` [Google
      IAM](https://cloud.google.com/iam/) permission on the Task resource.

  Fields:
    responseView: The response_view specifies which subset of the Task will be
      returned. By default response_view is BASIC; not all information is
      retrieved by default because some data, such as payloads, might be
      desirable to return only when needed because of its large size or
      because of the sensitivity of data that it contains. Authorization for
      FULL requires `cloudtasks.tasks.fullView` [Google
      IAM](https://cloud.google.com/iam/) permission on the Task resource.
    task: Required. The task to add. Task names have the following format:
      `projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID/tasks/TASK_ID
      `. The user can optionally specify a task name. If a name is not
      specified then the system will generate a random unique task id, which
      will be set in the task returned in the response. If schedule_time is
      not set or is in the past then Cloud Tasks will set it to the current
      time. Task De-duplication: Explicitly specifying a task ID enables task
      de-duplication. If a task's ID is identical to that of an existing task
      or a task that was deleted or executed recently then the call will fail
      with ALREADY_EXISTS. The IDs of deleted tasks are not immediately
      available for reuse. It can take up to 4 hours (or 9 days if the task's
      queue was created using a queue.yaml or queue.xml) for the task ID to be
      released and made available again. Because there is an extra lookup cost
      to identify duplicate task names, these CreateTask calls have
      significantly increased latency. Using hashed strings for the task id or
      for the prefix of the task id is recommended. Choosing task ids that are
      sequential or have sequential prefixes, for example using a timestamp,
      causes an increase in latency and error rates in all task commands. The
      infrastructure relies on an approximately uniform distribution of task
      ids to store and serve tasks efficiently.
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
    responseView = _messages.EnumField('ResponseViewValueValuesEnum', 1)
    task = _messages.MessageField('Task', 2)