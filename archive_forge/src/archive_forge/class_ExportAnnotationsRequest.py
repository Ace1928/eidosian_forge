from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportAnnotationsRequest(_messages.Message):
    """Request to export Annotations. The export operation is not atomic. If a
  failure occurs, any annotations already imported are not removed.

  Fields:
    bigqueryDestination: The BigQuery output destination, which requires two
      IAM roles: `roles/bigquery.dataEditor` and `roles/bigquery.jobUser`.
    gcsDestination: The Cloud Storage destination, which requires the
      `roles/storage.objectAdmin` Cloud IAM role.
  """
    bigqueryDestination = _messages.MessageField('GoogleCloudHealthcareV1alpha2AnnotationBigQueryDestination', 1)
    gcsDestination = _messages.MessageField('GoogleCloudHealthcareV1alpha2AnnotationGcsDestination', 2)