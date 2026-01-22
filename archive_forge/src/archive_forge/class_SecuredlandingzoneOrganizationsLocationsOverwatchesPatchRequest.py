from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuredlandingzoneOrganizationsLocationsOverwatchesPatchRequest(_messages.Message):
    """A SecuredlandingzoneOrganizationsLocationsOverwatchesPatchRequest
  object.

  Fields:
    googleCloudSecuredlandingzoneV1betaOverwatch: A
      GoogleCloudSecuredlandingzoneV1betaOverwatch resource to be passed as
      the request body.
    name: Output only. The name of this Overwatch resource, in the format of
      organizations/{org_id}/locations/{location_id}/overwatches/{overwatch_id
      }.
    updateMask: Optional. The FieldMask to use when updating the Overwatch.
      Only the fields specified here will be updated. This should not be
      empty. Updatable fields are: * blueprint_plan
  """
    googleCloudSecuredlandingzoneV1betaOverwatch = _messages.MessageField('GoogleCloudSecuredlandingzoneV1betaOverwatch', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)