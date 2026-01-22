from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1ListProcessorTypesResponse(_messages.Message):
    """Response message for the ListProcessorTypes method.

  Fields:
    nextPageToken: Points to the next page, otherwise empty.
    processorTypes: The processor types.
  """
    nextPageToken = _messages.StringField(1)
    processorTypes = _messages.MessageField('GoogleCloudDocumentaiV1ProcessorType', 2, repeated=True)