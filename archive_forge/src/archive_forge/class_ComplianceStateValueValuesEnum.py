from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComplianceStateValueValuesEnum(_messages.Enum):
    """The compliance state of the resource as specified by the API client.

    Values:
      COMPLIANCE_STATE_UNSPECIFIED: The compliance state of the resource is
        unknown or unspecified.
      COMPLIANT: Device is compliant with third party policies
      NON_COMPLIANT: Device is not compliant with third party policies
    """
    COMPLIANCE_STATE_UNSPECIFIED = 0
    COMPLIANT = 1
    NON_COMPLIANT = 2