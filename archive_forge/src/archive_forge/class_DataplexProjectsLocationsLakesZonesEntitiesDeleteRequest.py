from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesZonesEntitiesDeleteRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesZonesEntitiesDeleteRequest object.

  Fields:
    etag: Required. The etag associated with the entity, which can be
      retrieved with a GetEntity request.
    name: Required. The resource name of the entity: projects/{project_number}
      /locations/{location_id}/lakes/{lake_id}/zones/{zone_id}/entities/{entit
      y_id}.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)