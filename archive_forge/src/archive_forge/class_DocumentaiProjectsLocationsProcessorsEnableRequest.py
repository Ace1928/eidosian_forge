from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorsEnableRequest(_messages.Message):
    """A DocumentaiProjectsLocationsProcessorsEnableRequest object.

  Fields:
    googleCloudDocumentaiV1EnableProcessorRequest: A
      GoogleCloudDocumentaiV1EnableProcessorRequest resource to be passed as
      the request body.
    name: Required. The processor resource name to be enabled.
  """
    googleCloudDocumentaiV1EnableProcessorRequest = _messages.MessageField('GoogleCloudDocumentaiV1EnableProcessorRequest', 1)
    name = _messages.StringField(2, required=True)