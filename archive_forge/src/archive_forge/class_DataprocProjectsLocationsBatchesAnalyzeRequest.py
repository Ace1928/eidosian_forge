from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsBatchesAnalyzeRequest(_messages.Message):
    """A DataprocProjectsLocationsBatchesAnalyzeRequest object.

  Fields:
    analyzeBatchRequest: A AnalyzeBatchRequest resource to be passed as the
      request body.
    name: Required. The fully qualified name of the batch to analyze in the
      format "projects/PROJECT_ID/locations/DATAPROC_REGION/batches/BATCH_ID"
  """
    analyzeBatchRequest = _messages.MessageField('AnalyzeBatchRequest', 1)
    name = _messages.StringField(2, required=True)