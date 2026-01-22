from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PreviewResults(_messages.Message):
    """Locations of outputs from config preview.

  Fields:
    artifacts: Location of kpt artifacts in Google Cloud Storage. Format:
      `gs://{bucket}/{object}`
    content: Location of generated preview data in Google Cloud Storage.
      Format: `gs://{bucket}/{object}`
  """
    artifacts = _messages.StringField(1)
    content = _messages.StringField(2)