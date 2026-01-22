from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterProjectsSourcesFindingsExternalSystemsPatchRequest(_messages.Message):
    """A SecuritycenterProjectsSourcesFindingsExternalSystemsPatchRequest
  object.

  Fields:
    googleCloudSecuritycenterV2ExternalSystem: A
      GoogleCloudSecuritycenterV2ExternalSystem resource to be passed as the
      request body.
    name: Full resource name of the external system. The following list shows
      some examples: +
      `organizations/1234/sources/5678/findings/123456/externalSystems/jira` +
      `organizations/1234/sources/5678/locations/us/findings/123456/externalSy
      stems/jira` +
      `folders/1234/sources/5678/findings/123456/externalSystems/jira` + `fold
      ers/1234/sources/5678/locations/us/findings/123456/externalSystems/jira`
      + `projects/1234/sources/5678/findings/123456/externalSystems/jira` + `p
      rojects/1234/sources/5678/locations/us/findings/123456/externalSystems/j
      ira`
    updateMask: The FieldMask to use when updating the external system
      resource. If empty all mutable fields will be updated.
  """
    googleCloudSecuritycenterV2ExternalSystem = _messages.MessageField('GoogleCloudSecuritycenterV2ExternalSystem', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)