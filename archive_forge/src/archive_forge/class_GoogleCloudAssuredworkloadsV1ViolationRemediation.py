from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1ViolationRemediation(_messages.Message):
    """Represents remediation guidance to resolve compliance violation for
  AssuredWorkload

  Enums:
    RemediationTypeValueValuesEnum: Output only. Reemediation type based on
      the type of org policy values violated

  Fields:
    compliantValues: Values that can resolve the violation For example: for
      list org policy violations, this will either be the list of allowed or
      denied values
    instructions: Required. Remediation instructions to resolve violations
    remediationType: Output only. Reemediation type based on the type of org
      policy values violated
  """

    class RemediationTypeValueValuesEnum(_messages.Enum):
        """Output only. Reemediation type based on the type of org policy values
    violated

    Values:
      REMEDIATION_TYPE_UNSPECIFIED: Unspecified remediation type
      REMEDIATION_BOOLEAN_ORG_POLICY_VIOLATION: Remediation type for boolean
        org policy
      REMEDIATION_LIST_ALLOWED_VALUES_ORG_POLICY_VIOLATION: Remediation type
        for list org policy which have allowed values in the monitoring rule
      REMEDIATION_LIST_DENIED_VALUES_ORG_POLICY_VIOLATION: Remediation type
        for list org policy which have denied values in the monitoring rule
      REMEDIATION_RESTRICT_CMEK_CRYPTO_KEY_PROJECTS_ORG_POLICY_VIOLATION:
        Remediation type for gcp.restrictCmekCryptoKeyProjects
      REMEDIATION_RESOURCE_VIOLATION: Remediation type for resource violation.
    """
        REMEDIATION_TYPE_UNSPECIFIED = 0
        REMEDIATION_BOOLEAN_ORG_POLICY_VIOLATION = 1
        REMEDIATION_LIST_ALLOWED_VALUES_ORG_POLICY_VIOLATION = 2
        REMEDIATION_LIST_DENIED_VALUES_ORG_POLICY_VIOLATION = 3
        REMEDIATION_RESTRICT_CMEK_CRYPTO_KEY_PROJECTS_ORG_POLICY_VIOLATION = 4
        REMEDIATION_RESOURCE_VIOLATION = 5
    compliantValues = _messages.StringField(1, repeated=True)
    instructions = _messages.MessageField('GoogleCloudAssuredworkloadsV1ViolationRemediationInstructions', 2)
    remediationType = _messages.EnumField('RemediationTypeValueValuesEnum', 3)