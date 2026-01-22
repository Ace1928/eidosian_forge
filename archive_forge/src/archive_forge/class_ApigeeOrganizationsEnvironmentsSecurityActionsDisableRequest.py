from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSecurityActionsDisableRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSecurityActionsDisableRequest object.

  Fields:
    googleCloudApigeeV1DisableSecurityActionRequest: A
      GoogleCloudApigeeV1DisableSecurityActionRequest resource to be passed as
      the request body.
    name: Required. The name of the SecurityAction to disable. Format:
      organizations/{org}/environments/{env}/securityActions/{security_action}
  """
    googleCloudApigeeV1DisableSecurityActionRequest = _messages.MessageField('GoogleCloudApigeeV1DisableSecurityActionRequest', 1)
    name = _messages.StringField(2, required=True)