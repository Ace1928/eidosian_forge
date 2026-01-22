from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3ExportProcessorVersionResponse(_messages.Message):
    """Response message associated with the ExportProcessorVersion operation.

  Fields:
    gcsUri: The Cloud Storage URI containing the output artifacts.
  """
    gcsUri = _messages.StringField(1)