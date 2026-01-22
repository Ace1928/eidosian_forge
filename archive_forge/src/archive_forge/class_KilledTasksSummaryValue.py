from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class KilledTasksSummaryValue(_messages.Message):
    """A KilledTasksSummaryValue object.

    Messages:
      AdditionalProperty: An additional property for a KilledTasksSummaryValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        KilledTasksSummaryValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a KilledTasksSummaryValue object.

      Fields:
        key: Name of the additional property.
        value: A integer attribute.
      """
        key = _messages.StringField(1)
        value = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)