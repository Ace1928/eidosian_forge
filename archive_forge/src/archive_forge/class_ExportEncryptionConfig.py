from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportEncryptionConfig(_messages.Message):
    """Configuration for Encryption - e.g. CMEK.

  Fields:
    kmsKeyName: Required. Name of the CMEK key in KMS.
  """
    kmsKeyName = _messages.StringField(1)