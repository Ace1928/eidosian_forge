from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsGetEventThreatDetectionSettingsRequest(_messages.Message):
    """A SecuritycenterOrganizationsGetEventThreatDetectionSettingsRequest
  object.

  Fields:
    name: Required. The name of the EventThreatDetectionSettings to retrieve.
      Formats: * organizations/{organization}/eventThreatDetectionSettings *
      folders/{folder}/eventThreatDetectionSettings *
      projects/{project}/eventThreatDetectionSettings
  """
    name = _messages.StringField(1, required=True)