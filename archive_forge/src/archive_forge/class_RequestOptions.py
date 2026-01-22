from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RequestOptions(_messages.Message):
    """Common request options for various APIs.

  Enums:
    PriorityValueValuesEnum: Priority for the request.

  Fields:
    priority: Priority for the request.
    requestTag: A per-request tag which can be applied to queries or reads,
      used for statistics collection. Both request_tag and transaction_tag can
      be specified for a read or query that belongs to a transaction. This
      field is ignored for requests where it's not applicable (e.g.
      CommitRequest). Legal characters for `request_tag` values are all
      printable characters (ASCII 32 - 126) and the length of a request_tag is
      limited to 50 characters. Values that exceed this limit are truncated.
      Any leading underscore (_) characters will be removed from the string.
    transactionTag: A tag used for statistics collection about this
      transaction. Both request_tag and transaction_tag can be specified for a
      read or query that belongs to a transaction. The value of
      transaction_tag should be the same for all requests belonging to the
      same transaction. If this request doesn't belong to any transaction,
      transaction_tag will be ignored. Legal characters for `transaction_tag`
      values are all printable characters (ASCII 32 - 126) and the length of a
      transaction_tag is limited to 50 characters. Values that exceed this
      limit are truncated. Any leading underscore (_) characters will be
      removed from the string.
  """

    class PriorityValueValuesEnum(_messages.Enum):
        """Priority for the request.

    Values:
      PRIORITY_UNSPECIFIED: `PRIORITY_UNSPECIFIED` is equivalent to
        `PRIORITY_HIGH`.
      PRIORITY_LOW: This specifies that the request is low priority.
      PRIORITY_MEDIUM: This specifies that the request is medium priority.
      PRIORITY_HIGH: This specifies that the request is high priority.
    """
        PRIORITY_UNSPECIFIED = 0
        PRIORITY_LOW = 1
        PRIORITY_MEDIUM = 2
        PRIORITY_HIGH = 3
    priority = _messages.EnumField('PriorityValueValuesEnum', 1)
    requestTag = _messages.StringField(2)
    transactionTag = _messages.StringField(3)