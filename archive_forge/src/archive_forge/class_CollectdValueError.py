from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CollectdValueError(_messages.Message):
    """Describes the error status for values that were not written.

  Fields:
    error: Records the error status for the value.
    index: The zero-based index in CollectdPayload.values within the parent
      CreateCollectdTimeSeriesRequest.collectd_payloads.
  """
    error = _messages.MessageField('Status', 1)
    index = _messages.IntegerField(2, variant=_messages.Variant.INT32)