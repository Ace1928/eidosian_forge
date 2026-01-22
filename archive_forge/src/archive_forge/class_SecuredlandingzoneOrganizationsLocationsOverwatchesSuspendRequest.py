from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuredlandingzoneOrganizationsLocationsOverwatchesSuspendRequest(_messages.Message):
    """A SecuredlandingzoneOrganizationsLocationsOverwatchesSuspendRequest
  object.

  Fields:
    googleCloudSecuredlandingzoneV1betaSuspendOverwatchRequest: A
      GoogleCloudSecuredlandingzoneV1betaSuspendOverwatchRequest resource to
      be passed as the request body.
    name: Required. The name of the Overwatch resource to suspend. The format
      is organizations/{org_id}/locations/{location_id}/overwatches/{overwatch
      _id}.
  """
    googleCloudSecuredlandingzoneV1betaSuspendOverwatchRequest = _messages.MessageField('GoogleCloudSecuredlandingzoneV1betaSuspendOverwatchRequest', 1)
    name = _messages.StringField(2, required=True)