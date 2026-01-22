from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuredlandingzoneOrganizationsLocationsOverwatchesActivateRequest(_messages.Message):
    """A SecuredlandingzoneOrganizationsLocationsOverwatchesActivateRequest
  object.

  Fields:
    googleCloudSecuredlandingzoneV1betaActivateOverwatchRequest: A
      GoogleCloudSecuredlandingzoneV1betaActivateOverwatchRequest resource to
      be passed as the request body.
    name: Required. The name of the Overwatch resource to activate. The format
      is organizations/{org_id}/locations/{location_id}/overwatches/{overwatch
      _id}.
  """
    googleCloudSecuredlandingzoneV1betaActivateOverwatchRequest = _messages.MessageField('GoogleCloudSecuredlandingzoneV1betaActivateOverwatchRequest', 1)
    name = _messages.StringField(2, required=True)