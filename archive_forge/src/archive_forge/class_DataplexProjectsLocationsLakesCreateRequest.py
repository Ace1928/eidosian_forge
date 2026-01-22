from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesCreateRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesCreateRequest object.

  Fields:
    googleCloudDataplexV1Lake: A GoogleCloudDataplexV1Lake resource to be
      passed as the request body.
    lakeId: Required. Lake identifier. This ID will be used to generate names
      such as database and dataset names when publishing metadata to Hive
      Metastore and BigQuery. * Must contain only lowercase letters, numbers
      and hyphens. * Must start with a letter. * Must end with a number or a
      letter. * Must be between 1-63 characters. * Must be unique within the
      customer project / location.
    parent: Required. The resource name of the lake location, of the form:
      projects/{project_number}/locations/{location_id} where location_id
      refers to a GCP region.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    googleCloudDataplexV1Lake = _messages.MessageField('GoogleCloudDataplexV1Lake', 1)
    lakeId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)