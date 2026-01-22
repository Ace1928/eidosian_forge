from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class StragglerCauseCountValue(_messages.Message):
    """Aggregated counts of straggler causes, keyed by the string
    representation of the StragglerCause enum.

    Messages:
      AdditionalProperty: An additional property for a
        StragglerCauseCountValue object.

    Fields:
      additionalProperties: Additional properties of type
        StragglerCauseCountValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a StragglerCauseCountValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.IntegerField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)