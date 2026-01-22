from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsUpdateControlPlaneAccessRequest(_messages.Message):
    """A ApigeeOrganizationsUpdateControlPlaneAccessRequest object.

  Fields:
    googleCloudApigeeV1ControlPlaneAccess: A
      GoogleCloudApigeeV1ControlPlaneAccess resource to be passed as the
      request body.
    name: The resource name of the ControlPlaneAccess. Format:
      "organizations/{org}/controlPlaneAccess"
    updateMask: List of fields to be updated. Fields that can be updated:
      portal_disabled, release_channel, addon_config.
  """
    googleCloudApigeeV1ControlPlaneAccess = _messages.MessageField('GoogleCloudApigeeV1ControlPlaneAccess', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)