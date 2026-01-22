from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta1GcsSource(_messages.Message):
    """The Google Cloud Storage location where the input file will be read
  from.

  Fields:
    uri: A string attribute.
  """
    uri = _messages.StringField(1)