from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycentermanagementOrganizationsLocationsEventThreatDetectionCustomModulesPatchRequest(_messages.Message):
    """A SecuritycentermanagementOrganizationsLocationsEventThreatDetectionCust
  omModulesPatchRequest object.

  Fields:
    eventThreatDetectionCustomModule: A EventThreatDetectionCustomModule
      resource to be passed as the request body.
    name: Identifier. The resource name of the ETD custom module. Its format
      is: * "organizations/{organization}/locations/{location}/eventThreatDete
      ctionCustomModules/{event_threat_detection_custom_module}". * "folders/{
      folder}/locations/{location}/eventThreatDetectionCustomModules/{event_th
      reat_detection_custom_module}". * "projects/{project}/locations/{locatio
      n}/eventThreatDetectionCustomModules/{event_threat_detection_custom_modu
      le}".
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten in the EventThreatDetectionCustomModule resource by the
      update. The fields specified in the update_mask are relative to the
      resource, not the full request. A field will be overwritten if it is in
      the mask. If the user does not provide a mask then all fields will be
      overwritten.
    validateOnly: Optional. When set to true, only validations (including IAM
      checks) will done for the request (module will not be updated). An OK
      response indicates the request is valid while an error response
      indicates the request is invalid. Note that a subsequent request to
      actually update the module could still fail because 1. the state could
      have changed (e.g. IAM permission lost) or 2. A failure occurred while
      trying to update the module.
  """
    eventThreatDetectionCustomModule = _messages.MessageField('EventThreatDetectionCustomModule', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)