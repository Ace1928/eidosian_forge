from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SoftDeletePolicyValue(_messages.Message):
    """The bucket's soft delete policy, which defines the period of time that
    soft-deleted objects will be retained, and cannot be permanently deleted.

    Fields:
      effectiveTime: Server-determined value that indicates the time from
        which the policy, or one with a greater retention, was effective. This
        value is in RFC 3339 format.
      retentionDurationSeconds: The duration in seconds that soft-deleted
        objects in the bucket will be retained and cannot be permanently
        deleted.
    """
    effectiveTime = _message_types.DateTimeField(1)
    retentionDurationSeconds = _messages.IntegerField(2)