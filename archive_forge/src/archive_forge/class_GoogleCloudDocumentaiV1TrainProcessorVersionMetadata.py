from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1TrainProcessorVersionMetadata(_messages.Message):
    """The metadata that represents a processor version being created.

  Fields:
    commonMetadata: The basic metadata of the long-running operation.
    testDatasetValidation: The test dataset validation information.
    trainingDatasetValidation: The training dataset validation information.
  """
    commonMetadata = _messages.MessageField('GoogleCloudDocumentaiV1CommonOperationMetadata', 1)
    testDatasetValidation = _messages.MessageField('GoogleCloudDocumentaiV1TrainProcessorVersionMetadataDatasetValidation', 2)
    trainingDatasetValidation = _messages.MessageField('GoogleCloudDocumentaiV1TrainProcessorVersionMetadataDatasetValidation', 3)