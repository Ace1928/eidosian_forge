from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class RuleStatsValue(_messages.Message):
    """A map from rule_id to RuleStats.

    Messages:
      AdditionalProperty: An additional property for a RuleStatsValue object.

    Fields:
      additionalProperties: Additional properties of type RuleStatsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a RuleStatsValue object.

      Fields:
        key: Name of the additional property.
        value: A RuleStats attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('RuleStats', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)