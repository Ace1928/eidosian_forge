from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DateShiftConfig(_messages.Message):
    """Shifts dates by random number of days, with option to be consistent for
  the same context. See https://cloud.google.com/sensitive-data-
  protection/docs/concepts-date-shifting to learn more.

  Fields:
    context: Points to the field that contains the context, for example, an
      entity id. If set, must also set cryptoKey. If set, shift will be
      consistent for the given context.
    cryptoKey: Causes the shift to be computed based on this key and the
      context. This results in the same shift for the same context and
      crypto_key. If set, must also set context. Can only be applied to table
      items.
    lowerBoundDays: Required. For example, -5 means shift date to at most 5
      days back in the past.
    upperBoundDays: Required. Range of shift in days. Actual shift will be
      selected at random within this range (inclusive ends). Negative means
      shift to earlier in time. Must not be more than 365250 days (1000 years)
      each direction. For example, 3 means shift date to at most 3 days into
      the future.
  """
    context = _messages.MessageField('GooglePrivacyDlpV2FieldId', 1)
    cryptoKey = _messages.MessageField('GooglePrivacyDlpV2CryptoKey', 2)
    lowerBoundDays = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    upperBoundDays = _messages.IntegerField(4, variant=_messages.Variant.INT32)