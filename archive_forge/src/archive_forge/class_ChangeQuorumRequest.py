from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChangeQuorumRequest(_messages.Message):
    """The request for ChangeQuorum.

  Fields:
    etag: Optional. The etag is the hash of the QuorumInfo. The ChangeQuorum
      operation will only be performed if the etag matches that of the
      QuorumInfo in the current database resource. Otherwise the API will
      return an `ABORTED` error. The etag is used for optimistic concurrency
      control as a way to help prevent simultaneous change quorum requests
      that could create a race condition.
    name: Required. Name of the database in which to apply the ChangeQuorum.
      Values are of the form `projects//instances//databases/`.
    quorumType: Required. The type of this Quorum.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2)
    quorumType = _messages.MessageField('QuorumType', 3)