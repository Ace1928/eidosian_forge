from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3TrainProcessorVersionMetadataDatasetValidation(_messages.Message):
    """The dataset validation information. This includes any and all errors
  with documents and the dataset.

  Fields:
    datasetErrorCount: The total number of dataset errors.
    datasetErrors: Error information for the dataset as a whole. A maximum of
      10 dataset errors will be returned. A single dataset error is terminal
      for training.
    documentErrorCount: The total number of document errors.
    documentErrors: Error information pertaining to specific documents. A
      maximum of 10 document errors will be returned. Any document with errors
      will not be used throughout training.
  """
    datasetErrorCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    datasetErrors = _messages.MessageField('GoogleRpcStatus', 2, repeated=True)
    documentErrorCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    documentErrors = _messages.MessageField('GoogleRpcStatus', 4, repeated=True)