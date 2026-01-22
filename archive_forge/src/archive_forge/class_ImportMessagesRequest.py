from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportMessagesRequest(_messages.Message):
    """Request to import messages.

  Fields:
    gcsSource: Cloud Storage source data location and import configuration.
      The Cloud Healthcare Service Agent requires the
      `roles/storage.objectViewer` Cloud IAM roles on the Cloud Storage
      location.
  """
    gcsSource = _messages.MessageField('GcsSource', 1)