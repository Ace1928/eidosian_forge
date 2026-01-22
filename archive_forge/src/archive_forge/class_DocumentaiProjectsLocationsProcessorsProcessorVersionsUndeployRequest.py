from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorsProcessorVersionsUndeployRequest(_messages.Message):
    """A DocumentaiProjectsLocationsProcessorsProcessorVersionsUndeployRequest
  object.

  Fields:
    googleCloudDocumentaiV1UndeployProcessorVersionRequest: A
      GoogleCloudDocumentaiV1UndeployProcessorVersionRequest resource to be
      passed as the request body.
    name: Required. The processor version resource name to be undeployed.
  """
    googleCloudDocumentaiV1UndeployProcessorVersionRequest = _messages.MessageField('GoogleCloudDocumentaiV1UndeployProcessorVersionRequest', 1)
    name = _messages.StringField(2, required=True)