from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuditTrail(_messages.Message):
    """Audit trail for the access provided by this Grant.

  Fields:
    accessGrantTime: Output only. The time at which access was given.
    accessRemoveTime: Output only. The time at which system removed access.
      This could be because of an automatic expiry or because of a revocation.
      If unspecified then access hasn't been removed yet.
  """
    accessGrantTime = _messages.StringField(1)
    accessRemoveTime = _messages.StringField(2)