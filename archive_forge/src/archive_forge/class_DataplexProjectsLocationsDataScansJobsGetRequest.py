from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataScansJobsGetRequest(_messages.Message):
    """A DataplexProjectsLocationsDataScansJobsGetRequest object.

  Enums:
    ViewValueValuesEnum: Optional. Select the DataScanJob view to return.
      Defaults to BASIC.

  Fields:
    name: Required. The resource name of the DataScanJob: projects/{project}/l
      ocations/{location_id}/dataScans/{data_scan_id}/jobs/{data_scan_job_id}
      where project refers to a project_id or project_number and location_id
      refers to a GCP region.
    view: Optional. Select the DataScanJob view to return. Defaults to BASIC.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. Select the DataScanJob view to return. Defaults to BASIC.

    Values:
      DATA_SCAN_JOB_VIEW_UNSPECIFIED: The API will default to the BASIC view.
      BASIC: Basic view that does not include spec and result.
      FULL: Include everything.
    """
        DATA_SCAN_JOB_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)