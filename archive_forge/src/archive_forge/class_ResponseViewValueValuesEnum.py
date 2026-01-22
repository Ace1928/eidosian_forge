from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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