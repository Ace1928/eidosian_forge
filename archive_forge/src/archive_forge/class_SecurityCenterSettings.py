from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityCenterSettings(_messages.Message):
    """Resource capturing the settings for Security Center. Next ID: 12

  Fields:
    logSinkProject: The resource name of the project to send logs to. This
      project must be part of the organization this resource resides in. The
      format is `projects/{project_id}`. An empty value disables logging. This
      value is only referenced by services that support log sink. Please refer
      to the documentation for an updated list of compatible services. This
      may only be specified for organization level onboarding.
    name: The resource name of the SecurityCenterSettings. Format:
      organizations/{organization}/securityCenterSettings Format:
      folders/{folder}/securityCenterSettings Format:
      projects/{project}/securityCenterSettings
    onboardingTime: Output only. Timestamp of when the customer organization
      was onboarded to SCC.
    orgServiceAccount: Output only. The organization level service account to
      be used for security center components.
  """
    logSinkProject = _messages.StringField(1)
    name = _messages.StringField(2)
    onboardingTime = _messages.StringField(3)
    orgServiceAccount = _messages.StringField(4)