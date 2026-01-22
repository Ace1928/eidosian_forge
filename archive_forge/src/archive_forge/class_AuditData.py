from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuditData(_messages.Message):
    """Audit log information specific to Cloud IAM. This message is serialized
  as an `Any` type in the `ServiceData` message of an `AuditLog` message.

  Fields:
    policyDelta: Policy delta between the original policy and the newly set
      policy.
  """
    policyDelta = _messages.MessageField('PolicyDelta', 1)