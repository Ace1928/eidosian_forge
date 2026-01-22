from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsUpdateEventThreatDetectionSettingsRequest(_messages.Message):
    """A SecuritycenterOrganizationsUpdateEventThreatDetectionSettingsRequest
  object.

  Fields:
    eventThreatDetectionSettings: A EventThreatDetectionSettings resource to
      be passed as the request body.
    name: The resource name of the EventThreatDetectionSettings. Formats: *
      organizations/{organization}/eventThreatDetectionSettings *
      folders/{folder}/eventThreatDetectionSettings *
      projects/{project}/eventThreatDetectionSettings
    updateMask: The list of fields to be updated.
  """
    eventThreatDetectionSettings = _messages.MessageField('EventThreatDetectionSettings', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)