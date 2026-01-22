from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1TrainProcessorVersionRequest(_messages.Message):
    """Request message for the TrainProcessorVersion method.

  Fields:
    baseProcessorVersion: Optional. The processor version to use as a base for
      training. This processor version must be a child of `parent`. Format: `p
      rojects/{project}/locations/{location}/processors/{processor}/processorV
      ersions/{processorVersion}`.
    customDocumentExtractionOptions: Options to control Custom Document
      Extraction (CDE) Processor.
    documentSchema: Optional. The schema the processor version will be trained
      with.
    foundationModelTuningOptions: Options to control foundation model tuning
      of a processor.
    inputData: Optional. The input data used to train the ProcessorVersion.
    processorVersion: Required. The processor version to be created.
  """
    baseProcessorVersion = _messages.StringField(1)
    customDocumentExtractionOptions = _messages.MessageField('GoogleCloudDocumentaiV1TrainProcessorVersionRequestCustomDocumentExtractionOptions', 2)
    documentSchema = _messages.MessageField('GoogleCloudDocumentaiV1DocumentSchema', 3)
    foundationModelTuningOptions = _messages.MessageField('GoogleCloudDocumentaiV1TrainProcessorVersionRequestFoundationModelTuningOptions', 4)
    inputData = _messages.MessageField('GoogleCloudDocumentaiV1TrainProcessorVersionRequestInputData', 5)
    processorVersion = _messages.MessageField('GoogleCloudDocumentaiV1ProcessorVersion', 6)