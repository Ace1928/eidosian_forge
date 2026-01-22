from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRedisV1OperationMetadata(_messages.Message):
    """Represents the v1 metadata of the long-running operation.

  Fields:
    apiVersion: API version.
    cancelRequested: Specifies if cancellation was requested for the
      operation.
    createTime: Creation timestamp.
    endTime: End timestamp.
    statusDetail: Operation status details.
    target: Operation target.
    verb: Operation verb.
  """
    apiVersion = _messages.StringField(1)
    cancelRequested = _messages.BooleanField(2)
    createTime = _messages.StringField(3)
    endTime = _messages.StringField(4)
    statusDetail = _messages.StringField(5)
    target = _messages.StringField(6)
    verb = _messages.StringField(7)