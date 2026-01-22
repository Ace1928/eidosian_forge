from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsBatchesSparkApplicationsAccessStageAttemptRequest(_messages.Message):
    """A
  DataprocProjectsLocationsBatchesSparkApplicationsAccessStageAttemptRequest
  object.

  Fields:
    name: Required. The fully qualified name of the batch to retrieve in the
      format "projects/PROJECT_ID/locations/DATAPROC_REGION/batches/BATCH_ID/s
      parkApplications/APPLICATION_ID"
    parent: Required. Parent (Batch) resource reference.
    stageAttemptId: Required. Stage Attempt ID
    stageId: Required. Stage ID
    summaryMetricsMask: Optional. The list of summary metrics fields to
      include. Empty list will default to skip all summary metrics fields.
      Example, if the response should include TaskQuantileMetrics, the request
      should have task_quantile_metrics in summary_metrics_mask field
  """
    name = _messages.StringField(1, required=True)
    parent = _messages.StringField(2)
    stageAttemptId = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    stageId = _messages.IntegerField(4)
    summaryMetricsMask = _messages.StringField(5)