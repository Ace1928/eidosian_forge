from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsJobsExecutionsExportStatusRequest(_messages.Message):
    """A RunProjectsLocationsJobsExecutionsExportStatusRequest object.

  Fields:
    name: Required. The name of the resource of which image export operation
      status has to be fetched. Format: `projects/{project_id_or_number}/locat
      ions/{location}/services/{service}/revisions/{revision}` for Revision `p
      rojects/{project_id_or_number}/locations/{location}/jobs/{job}/execution
      s/{execution}` for Execution
    operationId: Required. The operation id returned from ExportImage.
  """
    name = _messages.StringField(1, required=True)
    operationId = _messages.StringField(2, required=True)