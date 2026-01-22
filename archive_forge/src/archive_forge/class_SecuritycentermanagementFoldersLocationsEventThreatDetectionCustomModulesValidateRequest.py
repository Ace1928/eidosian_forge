from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModulesValidateRequest(_messages.Message):
    """A SecuritycentermanagementFoldersLocationsEventThreatDetectionCustomModu
  lesValidateRequest object.

  Fields:
    parent: Required. Resource name of the parent to validate the Custom
      Module under. Its format is: *
      "organizations/{organization}/locations/{location}".
    validateEventThreatDetectionCustomModuleRequest: A
      ValidateEventThreatDetectionCustomModuleRequest resource to be passed as
      the request body.
  """
    parent = _messages.StringField(1, required=True)
    validateEventThreatDetectionCustomModuleRequest = _messages.MessageField('ValidateEventThreatDetectionCustomModuleRequest', 2)