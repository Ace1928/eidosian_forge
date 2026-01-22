from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3CreateVersionOperationMetadata(_messages.Message):
    """Metadata associated with the long running operation for
  Versions.CreateVersion.

  Fields:
    version: Name of the created version. Format:
      `projects//locations//agents//flows//versions/`.
  """
    version = _messages.StringField(1)