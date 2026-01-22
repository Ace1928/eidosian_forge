from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExternalServiceCost(_messages.Message):
    """The external service cost is a portion of the total cost, these costs
  are not additive with total_bytes_billed. Moreover, this field only track
  external service costs that will show up as BigQuery costs (e.g. training
  BigQuery ML job with google cloud CAIP or Automl Tables services), not other
  costs which may be accrued by running the query (e.g. reading from Bigtable
  or Cloud Storage). The external service costs with different billing sku
  (e.g. CAIP job is charged based on VM usage) are converted to BigQuery
  billed_bytes and slot_ms with equivalent amount of US dollars. Services may
  not directly correlate to these metrics, but these are the equivalents for
  billing purposes. Output only.

  Fields:
    bytesBilled: External service cost in terms of bigquery bytes billed.
    bytesProcessed: External service cost in terms of bigquery bytes
      processed.
    externalService: External service name.
    reservedSlotCount: Non-preemptable reserved slots used for external job.
      For example, reserved slots for Cloua AI Platform job are the VM usages
      converted to BigQuery slot with equivalent mount of price.
    slotMs: External service cost in terms of bigquery slot milliseconds.
  """
    bytesBilled = _messages.IntegerField(1)
    bytesProcessed = _messages.IntegerField(2)
    externalService = _messages.StringField(3)
    reservedSlotCount = _messages.IntegerField(4)
    slotMs = _messages.IntegerField(5)