from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IaCValidationReport(_messages.Message):
    """Details of an IaC Validation report.

  Fields:
    skippedPolicies: List of policies unsupported by evaluation services
      during IAC validation.
    violations: List of violations found in the provided IaC.
  """
    skippedPolicies = _messages.MessageField('GoogleCloudSecuritypostureV1alphaIaCValidationReportPolicyDetails', 1, repeated=True)
    violations = _messages.MessageField('Violation', 2, repeated=True)