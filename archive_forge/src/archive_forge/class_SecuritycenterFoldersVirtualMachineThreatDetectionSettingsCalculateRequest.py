from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersVirtualMachineThreatDetectionSettingsCalculateRequest(_messages.Message):
    """A
  SecuritycenterFoldersVirtualMachineThreatDetectionSettingsCalculateRequest
  object.

  Fields:
    name: Required. The name of the VirtualMachineThreatDetectionSettings to
      calculate. Formats: *
      organizations/{organization}/virtualMachineThreatDetectionSettings *
      folders/{folder}/virtualMachineThreatDetectionSettings *
      projects/{project}/virtualMachineThreatDetectionSettings
  """
    name = _messages.StringField(1, required=True)