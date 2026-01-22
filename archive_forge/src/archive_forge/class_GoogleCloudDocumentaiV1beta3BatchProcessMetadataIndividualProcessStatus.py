from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta3BatchProcessMetadataIndividualProcessStatus(_messages.Message):
    """The status of a each individual document in the batch process.

  Fields:
    humanReviewOperation: The name of the operation triggered by the processed
      document. If the human review process isn't triggered, this field will
      be empty. It has the same response type and metadata as the long-running
      operation returned by the ReviewDocument method.
    humanReviewStatus: The status of human review on the processed document.
    inputGcsSource: The source of the document, same as the input_gcs_source
      field in the request when the batch process started.
    outputGcsDestination: The Cloud Storage output destination (in the request
      as DocumentOutputConfig.GcsOutputConfig.gcs_uri) of the processed
      document if it was successful, otherwise empty.
    status: The status processing the document.
  """
    humanReviewOperation = _messages.StringField(1)
    humanReviewStatus = _messages.MessageField('GoogleCloudDocumentaiV1beta3HumanReviewStatus', 2)
    inputGcsSource = _messages.StringField(3)
    outputGcsDestination = _messages.StringField(4)
    status = _messages.MessageField('GoogleRpcStatus', 5)