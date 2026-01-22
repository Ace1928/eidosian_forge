from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataScansRunRequest(_messages.Message):
    """A DataplexProjectsLocationsDataScansRunRequest object.

  Fields:
    googleCloudDataplexV1RunDataScanRequest: A
      GoogleCloudDataplexV1RunDataScanRequest resource to be passed as the
      request body.
    name: Required. The resource name of the DataScan:
      projects/{project}/locations/{location_id}/dataScans/{data_scan_id}.
      where project refers to a project_id or project_number and location_id
      refers to a GCP region.Only OnDemand data scans are allowed.
  """
    googleCloudDataplexV1RunDataScanRequest = _messages.MessageField('GoogleCloudDataplexV1RunDataScanRequest', 1)
    name = _messages.StringField(2, required=True)