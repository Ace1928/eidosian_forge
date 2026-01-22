from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesZonesCreateRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesZonesCreateRequest object.

  Fields:
    googleCloudDataplexV1Zone: A GoogleCloudDataplexV1Zone resource to be
      passed as the request body.
    parent: Required. The resource name of the parent lake:
      projects/{project_number}/locations/{location_id}/lakes/{lake_id}.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
    zoneId: Required. Zone identifier. This ID will be used to generate names
      such as database and dataset names when publishing metadata to Hive
      Metastore and BigQuery. * Must contain only lowercase letters, numbers
      and hyphens. * Must start with a letter. * Must end with a number or a
      letter. * Must be between 1-63 characters. * Must be unique across all
      lakes from all locations in a project. * Must not be one of the reserved
      IDs (i.e. "default", "global-temp")
  """
    googleCloudDataplexV1Zone = _messages.MessageField('GoogleCloudDataplexV1Zone', 1)
    parent = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)
    zoneId = _messages.StringField(4)