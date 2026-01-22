from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IngestMessageRequest(_messages.Message):
    """Ingests a message into the specified HL7v2 store.

  Fields:
    message: HL7v2 message to ingest.
  """
    message = _messages.MessageField('Message', 1)