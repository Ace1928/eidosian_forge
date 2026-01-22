from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModulesGetRequest(_messages.Message):
    """A SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModu
  lesGetRequest object.

  Fields:
    name: Required. The resource name of the ETD custom module. Its format is:
      * "organizations/{organization}/locations/{location}/eventThreatDetectio
      nCustomModules/{event_threat_detection_custom_module}". * "folders/{fold
      er}/locations/{location}/eventThreatDetectionCustomModules/{event_threat
      _detection_custom_module}". * "projects/{project}/locations/{location}/e
      ventThreatDetectionCustomModules/{event_threat_detection_custom_module}"
      .
  """
    name = _messages.StringField(1, required=True)