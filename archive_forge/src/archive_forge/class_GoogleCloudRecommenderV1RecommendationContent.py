from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1RecommendationContent(_messages.Message):
    """Contains what resources are changing and how they are changing.

  Messages:
    OverviewValue: Condensed overview information about the recommendation.

  Fields:
    operationGroups: Operations to one or more Google Cloud resources grouped
      in such a way that, all operations within one group are expected to be
      performed atomically and in an order.
    overview: Condensed overview information about the recommendation.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class OverviewValue(_messages.Message):
        """Condensed overview information about the recommendation.

    Messages:
      AdditionalProperty: An additional property for a OverviewValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a OverviewValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    operationGroups = _messages.MessageField('GoogleCloudRecommenderV1OperationGroup', 1, repeated=True)
    overview = _messages.MessageField('OverviewValue', 2)