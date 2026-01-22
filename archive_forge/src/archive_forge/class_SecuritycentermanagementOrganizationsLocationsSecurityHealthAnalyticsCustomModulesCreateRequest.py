from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsCustomModulesCreateRequest(_messages.Message):
    """A SecuritycentermanagementOrganizationsLocationsSecurityHealthAnalyticsC
  ustomModulesCreateRequest object.

  Fields:
    parent: Required. Name of the parent for the module. Its format is
      "organizations/{organization}/locations/{location}",
      "folders/{folder}/locations/{location}", or
      "projects/{project}/locations/{location}"
    securityHealthAnalyticsCustomModule: A SecurityHealthAnalyticsCustomModule
      resource to be passed as the request body.
    validateOnly: Optional. When set to true, only validations (including IAM
      checks) will done for the request (no module will be created). An OK
      response indicates the request is valid while an error response
      indicates the request is invalid. Note that a subsequent request to
      actually create the module could still fail because: 1. the state could
      have changed (e.g. IAM permission lost) or 2. A failure occurred during
      creation of the module. Defaults to false.
  """
    parent = _messages.StringField(1, required=True)
    securityHealthAnalyticsCustomModule = _messages.MessageField('SecurityHealthAnalyticsCustomModule', 2)
    validateOnly = _messages.BooleanField(3)