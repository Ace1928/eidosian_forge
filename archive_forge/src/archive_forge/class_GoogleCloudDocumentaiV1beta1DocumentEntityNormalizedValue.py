from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta1DocumentEntityNormalizedValue(_messages.Message):
    """Parsed and normalized entity value.

  Fields:
    addressValue: Postal address. See also: https://github.com/googleapis/goog
      leapis/blob/master/google/type/postal_address.proto
    booleanValue: Boolean value. Can be used for entities with binary values,
      or for checkboxes.
    dateValue: Date value. Includes year, month, day. See also: https://github
      .com/googleapis/googleapis/blob/master/google/type/date.proto
    datetimeValue: DateTime value. Includes date, time, and timezone. See
      also: https://github.com/googleapis/googleapis/blob/master/google/type/d
      atetime.proto
    floatValue: Float value.
    integerValue: Integer value.
    moneyValue: Money value. See also: https://github.com/googleapis/googleapi
      s/blob/master/google/type/money.proto
    text: Optional. An optional field to store a normalized string. For some
      entity types, one of respective `structured_value` fields may also be
      populated. Also not all the types of `structured_value` will be
      normalized. For example, some processors may not generate `float` or
      `integer` normalized text by default. Below are sample formats mapped to
      structured values. - Money/Currency type (`money_value`) is in the ISO
      4217 text format. - Date type (`date_value`) is in the ISO 8601 text
      format. - Datetime type (`datetime_value`) is in the ISO 8601 text
      format.
  """
    addressValue = _messages.MessageField('GoogleTypePostalAddress', 1)
    booleanValue = _messages.BooleanField(2)
    dateValue = _messages.MessageField('GoogleTypeDate', 3)
    datetimeValue = _messages.MessageField('GoogleTypeDateTime', 4)
    floatValue = _messages.FloatField(5, variant=_messages.Variant.FLOAT)
    integerValue = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    moneyValue = _messages.MessageField('GoogleTypeMoney', 7)
    text = _messages.StringField(8)