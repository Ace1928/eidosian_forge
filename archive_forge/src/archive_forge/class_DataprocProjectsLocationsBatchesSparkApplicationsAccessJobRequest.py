from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsBatchesSparkApplicationsAccessJobRequest(_messages.Message):
    """A DataprocProjectsLocationsBatchesSparkApplicationsAccessJobRequest
  object.

  Fields:
    jobId: Required. Job ID to fetch data for.
    name: Required. The fully qualified name of the batch to retrieve in the
      format "projects/PROJECT_ID/locations/DATAPROC_REGION/batches/BATCH_ID/s
      parkApplications/APPLICATION_ID"
    parent: Required. Parent (Batch) resource reference.
  """
    jobId = _messages.IntegerField(1)
    name = _messages.StringField(2, required=True)
    parent = _messages.StringField(3)