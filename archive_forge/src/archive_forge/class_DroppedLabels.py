from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DroppedLabels(_messages.Message):
    """A set of (label, value) pairs that were removed from a Distribution time
  series during aggregation and then added as an attachment to a
  Distribution.Exemplar.The full label set for the exemplars is constructed by
  using the dropped pairs in combination with the label values that remain on
  the aggregated Distribution time series. The constructed full label set can
  be used to identify the specific entity, such as the instance or job, which
  might be contributing to a long-tail. However, with dropped labels, the
  storage requirements are reduced because only the aggregated distribution
  values for a large group of time series are stored.Note that there are no
  guarantees on ordering of the labels from exemplar-to-exemplar and from
  distribution-to-distribution in the same stream, and there may be
  duplicates. It is up to clients to resolve any ambiguities.

  Messages:
    LabelValue: Map from label to its value, for all labels dropped in any
      aggregation.

  Fields:
    label: Map from label to its value, for all labels dropped in any
      aggregation.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelValue(_messages.Message):
        """Map from label to its value, for all labels dropped in any
    aggregation.

    Messages:
      AdditionalProperty: An additional property for a LabelValue object.

    Fields:
      additionalProperties: Additional properties of type LabelValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    label = _messages.MessageField('LabelValue', 1)