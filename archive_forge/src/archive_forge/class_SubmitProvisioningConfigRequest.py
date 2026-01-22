from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubmitProvisioningConfigRequest(_messages.Message):
    """Request for SubmitProvisioningConfig.

  Fields:
    email: Optional. Email provided to send a confirmation with provisioning
      config to.
    provisioningConfig: Required. The ProvisioningConfig to create.
  """
    email = _messages.StringField(1)
    provisioningConfig = _messages.MessageField('ProvisioningConfig', 2)