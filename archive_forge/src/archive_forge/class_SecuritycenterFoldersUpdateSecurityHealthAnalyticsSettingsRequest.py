from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersUpdateSecurityHealthAnalyticsSettingsRequest(_messages.Message):
    """A SecuritycenterFoldersUpdateSecurityHealthAnalyticsSettingsRequest
  object.

  Fields:
    name: The resource name of the SecurityHealthAnalyticsSettings. Formats: *
      organizations/{organization}/securityHealthAnalyticsSettings *
      folders/{folder}/securityHealthAnalyticsSettings *
      projects/{project}/securityHealthAnalyticsSettings
    securityHealthAnalyticsSettings: A SecurityHealthAnalyticsSettings
      resource to be passed as the request body.
    updateMask: The list of fields to be updated.
  """
    name = _messages.StringField(1, required=True)
    securityHealthAnalyticsSettings = _messages.MessageField('SecurityHealthAnalyticsSettings', 2)
    updateMask = _messages.StringField(3)