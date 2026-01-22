from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleRpcPreconditionFailureViolation(_messages.Message):
    """A message type used to describe a single precondition failure.

  Fields:
    description: A description of how the precondition failed. Developers can
      use this description to understand how to fix the failure. For example:
      "Terms of service not accepted".
    subject: The subject, relative to the type, that failed. For example,
      "google.com/cloud" relative to the "TOS" type would indicate which terms
      of service is being referenced.
    type: The type of PreconditionFailure. We recommend using a service-
      specific enum type to define the supported precondition violation
      subjects. For example, "TOS" for "Terms of Service violation".
  """
    description = _messages.StringField(1)
    subject = _messages.StringField(2)
    type = _messages.StringField(3)