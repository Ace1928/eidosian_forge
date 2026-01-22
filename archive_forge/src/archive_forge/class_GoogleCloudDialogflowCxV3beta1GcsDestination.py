from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1GcsDestination(_messages.Message):
    """Google Cloud Storage location for a Dialogflow operation that writes or
  exports objects (e.g. exported agent or transcripts) outside of Dialogflow.

  Fields:
    uri: Required. The Google Cloud Storage URI for the exported objects. A
      URI is of the form: `gs://bucket/object-name-or-prefix` Whether a full
      object name, or just a prefix, its usage depends on the Dialogflow
      operation.
  """
    uri = _messages.StringField(1)