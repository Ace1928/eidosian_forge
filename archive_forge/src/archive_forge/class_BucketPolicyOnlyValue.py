from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BucketPolicyOnlyValue(_messages.Message):
    """The bucket's Bucket Policy Only configuration.

      Fields:
        enabled: If set, access checks only use bucket-level IAM policies or
          above.
        lockedTime: The deadline time for changing
          iamConfiguration.bucketPolicyOnly.enabled from true to false in RFC
          3339 format. iamConfiguration.bucketPolicyOnly.enabled may be
          changed from true to false until the locked time, after which the
          field is immutable.
      """
    enabled = _messages.BooleanField(1)
    lockedTime = _message_types.DateTimeField(2)