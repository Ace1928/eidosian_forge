from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3SampleDocumentsResponseSelectedDocument(_messages.Message):
    """A GoogleCloudDocumentaiUiv1beta3SampleDocumentsResponseSelectedDocument
  object.

  Fields:
    documentId: An internal identifier for document.
  """
    documentId = _messages.StringField(1)