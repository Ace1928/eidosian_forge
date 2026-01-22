from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorsBatchProcessRequest(_messages.Message):
    """A DocumentaiProjectsLocationsProcessorsBatchProcessRequest object.

  Fields:
    googleCloudDocumentaiV1BatchProcessRequest: A
      GoogleCloudDocumentaiV1BatchProcessRequest resource to be passed as the
      request body.
    name: Required. The resource name of Processor or ProcessorVersion.
      Format:
      `projects/{project}/locations/{location}/processors/{processor}`, or `pr
      ojects/{project}/locations/{location}/processors/{processor}/processorVe
      rsions/{processorVersion}`
  """
    googleCloudDocumentaiV1BatchProcessRequest = _messages.MessageField('GoogleCloudDocumentaiV1BatchProcessRequest', 1)
    name = _messages.StringField(2, required=True)