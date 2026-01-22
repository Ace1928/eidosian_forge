from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportMessagesErrorDetails(_messages.Message):
    """Error response of importing messages. This structure is included in the
  error details to describe the detailed error. It is only included when the
  operation finishes with some failure.

  Fields:
    errorCount: The number of messages that had errors.
    hl7v2Store: The name of the target HL7v2 store, in the format `projects/{p
      roject_id}/locations/{location_id}/datasets/{dataset_id}/hl7v2Stores/{hl
      7v2_store_id}`
    inputSize: The total number of messages included in the source data. This
      is the sum of the success and error counts.
    successCount: The number of messages that have been imported.
  """
    errorCount = _messages.IntegerField(1)
    hl7v2Store = _messages.StringField(2)
    inputSize = _messages.IntegerField(3)
    successCount = _messages.IntegerField(4)