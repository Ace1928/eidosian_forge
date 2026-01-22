from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3AutoLabelDocumentsMetadata(_messages.Message):
    """Metadata of the auto-labeling documents operation.

  Fields:
    commonMetadata: The basic metadata of the long-running operation.
    individualAutoLabelStatuses: The list of individual auto-labeling statuses
      of the dataset documents.
    totalDocumentCount: Total number of the auto-labeling documents.
  """
    commonMetadata = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3CommonOperationMetadata', 1)
    individualAutoLabelStatuses = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3AutoLabelDocumentsMetadataIndividualAutoLabelStatus', 2, repeated=True)
    totalDocumentCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)