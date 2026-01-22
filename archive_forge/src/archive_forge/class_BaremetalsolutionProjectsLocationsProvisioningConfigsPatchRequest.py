from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsProvisioningConfigsPatchRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsProvisioningConfigsPatchRequest
  object.

  Fields:
    email: Optional. Email provided to send a confirmation with provisioning
      config to.
    name: Output only. The system-generated name of the provisioning config.
      This follows the UUID format.
    provisioningConfig: A ProvisioningConfig resource to be passed as the
      request body.
    updateMask: Required. The list of fields to update.
  """
    email = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    provisioningConfig = _messages.MessageField('ProvisioningConfig', 3)
    updateMask = _messages.StringField(4)