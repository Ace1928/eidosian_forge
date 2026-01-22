from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleRpcPreconditionFailure(_messages.Message):
    """Describes what preconditions have failed. For example, if an RPC failed
  because it required the Terms of Service to be acknowledged, it could list
  the terms of service violation in the PreconditionFailure message.

  Fields:
    violations: Describes all precondition violations.
  """
    violations = _messages.MessageField('GoogleRpcPreconditionFailureViolation', 1, repeated=True)