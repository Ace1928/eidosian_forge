from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PricingInfo(_messages.Message):
    """Represents the pricing information for a SKU at a single point of time.

  Fields:
    aggregationInfo: Aggregation Info. This can be left unspecified if the
      pricing expression doesn't require aggregation.
    currencyConversionRate: Conversion rate used for currency conversion, from
      USD to the currency specified in the request. This includes any
      surcharge collected for billing in non USD currency. If a currency is
      not specified in the request this defaults to 1.0. Example: USD *
      currency_conversion_rate = JPY
    effectiveTime: The timestamp from which this pricing was effective within
      the requested time range. This is guaranteed to be greater than or equal
      to the start_time field in the request and less than the end_time field
      in the request. If a time range was not specified in the request this
      field will be equivalent to a time within the last 12 hours, indicating
      the latest pricing info.
    pricingExpression: Expresses the pricing formula. See `PricingExpression`
      for an example.
    summary: An optional human readable summary of the pricing information,
      has a maximum length of 256 characters.
  """
    aggregationInfo = _messages.MessageField('AggregationInfo', 1)
    currencyConversionRate = _messages.FloatField(2)
    effectiveTime = _messages.StringField(3)
    pricingExpression = _messages.MessageField('PricingExpression', 4)
    summary = _messages.StringField(5)