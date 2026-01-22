from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidationStateValueValuesEnum(_messages.Enum):
    """Output only. Validation state of this certificate.

    Values:
      CERTIFICATE_VALIDATION_STATE_UNSPECIFIED: Default value.
      VALIDATION_SUCCESSFUL: Certificate validation was successful.
      VALIDATION_FAILED: Certificate validation failed.
    """
    CERTIFICATE_VALIDATION_STATE_UNSPECIFIED = 0
    VALIDATION_SUCCESSFUL = 1
    VALIDATION_FAILED = 2