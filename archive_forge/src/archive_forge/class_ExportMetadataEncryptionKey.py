from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportMetadataEncryptionKey(_messages.Message):
    """Encryption key details for the exported artifact.

  Fields:
    cmek: Name of the CMEK.
    version: Version of the CMEK.
  """
    cmek = _messages.StringField(1)
    version = _messages.StringField(2)