from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecognitionResponseMetadata(_messages.Message):
    """Metadata about the recognition request and response. Next ID: 10

  Fields:
    totalBilledDuration: When available, billed audio seconds for the
      corresponding request.
  """
    totalBilledDuration = _messages.StringField(1)