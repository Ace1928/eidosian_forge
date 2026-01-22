from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesCreateRequest(_messages.Message):
    """A
  SecuritycenterFoldersEventThreatDetectionSettingsCustomModulesCreateRequest
  object.

  Fields:
    eventThreatDetectionCustomModule: A EventThreatDetectionCustomModule
      resource to be passed as the request body.
    parent: Required. The new custom module's parent. Its format is: *
      "organizations/{organization}/eventThreatDetectionSettings". *
      "folders/{folder}/eventThreatDetectionSettings". *
      "projects/{project}/eventThreatDetectionSettings".
  """
    eventThreatDetectionCustomModule = _messages.MessageField('EventThreatDetectionCustomModule', 1)
    parent = _messages.StringField(2, required=True)