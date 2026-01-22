from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3CreateLabelerPoolOperationMetadata(_messages.Message):
    """The long-running operation metadata for the CreateLabelerPool method.

  Fields:
    commonMetadata: The basic metadata of the long-running operation.
  """
    commonMetadata = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3CommonOperationMetadata', 1)