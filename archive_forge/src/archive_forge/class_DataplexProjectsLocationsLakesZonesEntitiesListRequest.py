from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesZonesEntitiesListRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesZonesEntitiesListRequest object.

  Enums:
    ViewValueValuesEnum: Required. Specify the entity view to make a partial
      list request.

  Fields:
    filter: Optional. The following filter parameters can be added to the URL
      to limit the entities returned by the API: Entity ID:
      ?filter="id=entityID" Asset ID: ?filter="asset=assetID" Data path
      ?filter="data_path=gs://my-bucket" Is HIVE compatible:
      ?filter="hive_compatible=true" Is BigQuery compatible:
      ?filter="bigquery_compatible=true"
    pageSize: Optional. Maximum number of entities to return. The service may
      return fewer than this value. If unspecified, 100 entities will be
      returned by default. The maximum value is 500; larger values will will
      be truncated to 500.
    pageToken: Optional. Page token received from a previous ListEntities
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to ListEntities must match the call that
      provided the page token.
    parent: Required. The resource name of the parent zone: projects/{project_
      number}/locations/{location_id}/lakes/{lake_id}/zones/{zone_id}.
    view: Required. Specify the entity view to make a partial list request.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Required. Specify the entity view to make a partial list request.

    Values:
      ENTITY_VIEW_UNSPECIFIED: The default unset value. Return both table and
        fileset entities if unspecified.
      TABLES: Only list table entities.
      FILESETS: Only list fileset entities.
    """
        ENTITY_VIEW_UNSPECIFIED = 0
        TABLES = 1
        FILESETS = 2
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 5)