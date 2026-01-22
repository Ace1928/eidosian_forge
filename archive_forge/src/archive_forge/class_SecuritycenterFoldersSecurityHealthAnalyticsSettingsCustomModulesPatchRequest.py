from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersSecurityHealthAnalyticsSettingsCustomModulesPatchRequest(_messages.Message):
    """A SecuritycenterFoldersSecurityHealthAnalyticsSettingsCustomModulesPatch
  Request object.

  Fields:
    googleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule: A
      GoogleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule resource
      to be passed as the request body.
    name: Immutable. The resource name of the custom module. Its format is "or
      ganizations/{organization}/securityHealthAnalyticsSettings/customModules
      /{customModule}", or "folders/{folder}/securityHealthAnalyticsSettings/c
      ustomModules/{customModule}", or "projects/{project}/securityHealthAnaly
      ticsSettings/customModules/{customModule}" The id {customModule} is
      server-generated and is not user settable. It will be a numeric id
      containing 1-20 digits.
    updateMask: The list of fields to be updated. The only fields that can be
      updated are `enablement_state` and `custom_config`. If empty or set to
      the wildcard value `*`, both `enablement_state` and `custom_config` are
      updated.
  """
    googleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule = _messages.MessageField('GoogleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)