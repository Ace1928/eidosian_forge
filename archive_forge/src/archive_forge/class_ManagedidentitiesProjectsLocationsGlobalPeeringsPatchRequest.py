from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalPeeringsPatchRequest(_messages.Message):
    """A ManagedidentitiesProjectsLocationsGlobalPeeringsPatchRequest object.

  Fields:
    name: Output only. Unique name of the peering in this scope including
      projects and location using the form:
      `projects/{project_id}/locations/global/peerings/{peering_id}`.
    peering: A Peering resource to be passed as the request body.
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field. The elements of the repeated paths field may
      only include these fields from Peering: * `labels`
  """
    name = _messages.StringField(1, required=True)
    peering = _messages.MessageField('Peering', 2)
    updateMask = _messages.StringField(3)