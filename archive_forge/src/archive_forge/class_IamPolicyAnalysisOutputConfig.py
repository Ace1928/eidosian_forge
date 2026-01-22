from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamPolicyAnalysisOutputConfig(_messages.Message):
    """Output configuration for export IAM policy analysis destination.

  Fields:
    bigqueryDestination: Destination on BigQuery.
    gcsDestination: Destination on Cloud Storage.
  """
    bigqueryDestination = _messages.MessageField('GoogleCloudAssetV1BigQueryDestination', 1)
    gcsDestination = _messages.MessageField('GoogleCloudAssetV1GcsDestination', 2)