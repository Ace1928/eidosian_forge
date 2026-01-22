from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV3alpha1DeleteDocumentOperationMetadata(_messages.Message):
    """Metadata for DeleteDocument operation.

  Fields:
    genericMetadata: The generic information of the operation.
  """
    genericMetadata = _messages.MessageField('GoogleCloudDialogflowV3alpha1GenericKnowledgeOperationMetadata', 1)