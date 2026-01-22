from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuotaExceededInfo(_messages.Message):
    """Additional details for quota exceeded error for resource quota.

  Enums:
    RolloutStatusValueValuesEnum: Rollout status of the future quota limit.

  Messages:
    DimensionsValue: The map holding related quota dimensions.

  Fields:
    dimensions: The map holding related quota dimensions.
    futureLimit: Future quota limit being rolled out. The limit's unit depends
      on the quota type or metric.
    limit: Current effective quota limit. The limit's unit depends on the
      quota type or metric.
    limitName: The name of the quota limit.
    metricName: The Compute Engine quota metric name.
    rolloutStatus: Rollout status of the future quota limit.
  """

    class RolloutStatusValueValuesEnum(_messages.Enum):
        """Rollout status of the future quota limit.

    Values:
      IN_PROGRESS: IN_PROGRESS - A rollout is in process which will change the
        limit value to future limit.
      ROLLOUT_STATUS_UNSPECIFIED: ROLLOUT_STATUS_UNSPECIFIED - Rollout status
        is not specified. The default value.
    """
        IN_PROGRESS = 0
        ROLLOUT_STATUS_UNSPECIFIED = 1

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DimensionsValue(_messages.Message):
        """The map holding related quota dimensions.

    Messages:
      AdditionalProperty: An additional property for a DimensionsValue object.

    Fields:
      additionalProperties: Additional properties of type DimensionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DimensionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    dimensions = _messages.MessageField('DimensionsValue', 1)
    futureLimit = _messages.FloatField(2)
    limit = _messages.FloatField(3)
    limitName = _messages.StringField(4)
    metricName = _messages.StringField(5)
    rolloutStatus = _messages.EnumField('RolloutStatusValueValuesEnum', 6)