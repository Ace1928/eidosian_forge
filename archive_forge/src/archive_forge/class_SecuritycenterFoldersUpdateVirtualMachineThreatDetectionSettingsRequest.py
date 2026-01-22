from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersUpdateVirtualMachineThreatDetectionSettingsRequest(_messages.Message):
    """A
  SecuritycenterFoldersUpdateVirtualMachineThreatDetectionSettingsRequest
  object.

  Fields:
    name: The resource name of the VirtualMachineThreatDetectionSettings.
      Formats: *
      organizations/{organization}/virtualMachineThreatDetectionSettings *
      folders/{folder}/virtualMachineThreatDetectionSettings *
      projects/{project}/virtualMachineThreatDetectionSettings
    updateMask: The list of fields to be updated.
    virtualMachineThreatDetectionSettings: A
      VirtualMachineThreatDetectionSettings resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    virtualMachineThreatDetectionSettings = _messages.MessageField('VirtualMachineThreatDetectionSettings', 3)