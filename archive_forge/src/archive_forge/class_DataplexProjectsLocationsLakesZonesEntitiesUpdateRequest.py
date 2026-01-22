from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesZonesEntitiesUpdateRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesZonesEntitiesUpdateRequest object.

  Fields:
    googleCloudDataplexV1Entity: A GoogleCloudDataplexV1Entity resource to be
      passed as the request body.
    name: Output only. The resource name of the entity, of the form: projects/
      {project_number}/locations/{location_id}/lakes/{lake_id}/zones/{zone_id}
      /entities/{id}.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    googleCloudDataplexV1Entity = _messages.MessageField('GoogleCloudDataplexV1Entity', 1)
    name = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)