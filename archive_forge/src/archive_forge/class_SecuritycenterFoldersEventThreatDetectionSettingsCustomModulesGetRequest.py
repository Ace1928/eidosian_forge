from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesGetRequest(_messages.Message):
    """A
  SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesGetRequest
  object.

  Fields:
    name: Required. Name of the custom module to get. Its format is: * "organi
      zations/{organization}/eventThreatDetectionSettings/customModules/{modul
      e}". *
      "folders/{folder}/eventThreatDetectionSettings/customModules/{module}".
      * "projects/{project}/eventThreatDetectionSettings/customModules/{module
      }".
  """
    name = _messages.StringField(1, required=True)