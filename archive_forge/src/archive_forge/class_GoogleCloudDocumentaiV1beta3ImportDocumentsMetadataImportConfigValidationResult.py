from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta3ImportDocumentsMetadataImportConfigValidationResult(_messages.Message):
    """The validation status of each import config. Status is set to an error
  if there are no documents to import in the `import_config`, or `OK` if the
  operation will try to proceed with at least one document.

  Fields:
    inputGcsSource: The source Cloud Storage URI specified in the import
      config.
    status: The validation status of import config.
  """
    inputGcsSource = _messages.StringField(1)
    status = _messages.MessageField('GoogleRpcStatus', 2)