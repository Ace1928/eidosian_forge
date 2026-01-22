from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuredlandingzoneOrganizationsLocationsOverwatchesCreateRequest(_messages.Message):
    """A SecuredlandingzoneOrganizationsLocationsOverwatchesCreateRequest
  object.

  Fields:
    googleCloudSecuredlandingzoneV1betaOverwatch: A
      GoogleCloudSecuredlandingzoneV1betaOverwatch resource to be passed as
      the request body.
    overwatchId: Required. Unique id per organization per region for this
      overwatch instance.
    parent: Required. The name of the organization and region in which to
      create the Overwatch resource. The format is
      organizations/{org_id}/locations/{location_id}.
  """
    googleCloudSecuredlandingzoneV1betaOverwatch = _messages.MessageField('GoogleCloudSecuredlandingzoneV1betaOverwatch', 1)
    overwatchId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)