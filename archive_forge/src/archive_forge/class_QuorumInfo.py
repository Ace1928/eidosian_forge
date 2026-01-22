from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuorumInfo(_messages.Message):
    """Information about the dual region quorum.

  Enums:
    InitiatorValueValuesEnum: Output only. Whether this ChangeQuorum is a
      Google or User initiated.

  Fields:
    etag: Output only. The etag is used for optimistic concurrency control as
      a way to help prevent simultaneous ChangeQuorum requests that could
      create a race condition.
    initiator: Output only. Whether this ChangeQuorum is a Google or User
      initiated.
    quorumType: Output only. The type of this quorum. See QuorumType for more
      information about quorum type specifications.
    startTime: Output only. The timestamp when the request was triggered.
  """

    class InitiatorValueValuesEnum(_messages.Enum):
        """Output only. Whether this ChangeQuorum is a Google or User initiated.

    Values:
      INITIATOR_UNSPECIFIED: Unspecified.
      GOOGLE: ChangeQuorum initiated by Google.
      USER: ChangeQuorum initiated by User.
    """
        INITIATOR_UNSPECIFIED = 0
        GOOGLE = 1
        USER = 2
    etag = _messages.StringField(1)
    initiator = _messages.EnumField('InitiatorValueValuesEnum', 2)
    quorumType = _messages.MessageField('QuorumType', 3)
    startTime = _messages.StringField(4)