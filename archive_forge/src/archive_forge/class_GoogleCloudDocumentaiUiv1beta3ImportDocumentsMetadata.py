from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3ImportDocumentsMetadata(_messages.Message):
    """Metadata of the import document operation.

  Fields:
    commonMetadata: The basic metadata of the long-running operation.
    importConfigValidationResults: Validation statuses of the batch documents
      import config.
    individualImportStatuses: The list of response details of each document.
    totalDocumentCount: Total number of the documents that are qualified for
      importing.
  """
    commonMetadata = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3CommonOperationMetadata', 1)
    importConfigValidationResults = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3ImportDocumentsMetadataImportConfigValidationResult', 2, repeated=True)
    individualImportStatuses = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3ImportDocumentsMetadataIndividualImportStatus', 3, repeated=True)
    totalDocumentCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)