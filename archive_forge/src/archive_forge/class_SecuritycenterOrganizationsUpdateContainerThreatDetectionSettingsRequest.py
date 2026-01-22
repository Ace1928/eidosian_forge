from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsUpdateContainerThreatDetectionSettingsRequest(_messages.Message):
    """A
  SecuritycenterOrganizationsUpdateContainerThreatDetectionSettingsRequest
  object.

  Fields:
    containerThreatDetectionSettings: A ContainerThreatDetectionSettings
      resource to be passed as the request body.
    name: The resource name of the ContainerThreatDetectionSettings. Formats:
      * organizations/{organization}/containerThreatDetectionSettings *
      folders/{folder}/containerThreatDetectionSettings *
      projects/{project}/containerThreatDetectionSettings * projects/{project}
      /locations/{location}/clusters/{cluster}/containerThreatDetectionSetting
      s
    updateMask: The list of fields to be updated.
  """
    containerThreatDetectionSettings = _messages.MessageField('ContainerThreatDetectionSettings', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)