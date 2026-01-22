from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsEventThreatDetectionSettingsCustomModulesDeleteRequest(_messages.Message):
    """A SecuritycenterOrganizationsEventThreatDetectionSettingsCustomModulesDe
  leteRequest object.

  Fields:
    name: Required. Name of the custom module to delete. Its format is: * "org
      anizations/{organization}/eventThreatDetectionSettings/customModules/{mo
      dule}". *
      "folders/{folder}/eventThreatDetectionSettings/customModules/{module}".
      * "projects/{project}/eventThreatDetectionSettings/customModules/{module
      }".
  """
    name = _messages.StringField(1, required=True)