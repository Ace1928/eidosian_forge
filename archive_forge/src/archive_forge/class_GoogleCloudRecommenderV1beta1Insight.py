from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1beta1Insight(_messages.Message):
    """An insight along with the information used to derive the insight. The
  insight may have associated recommendations as well.

  Enums:
    CategoryValueValuesEnum: Category being targeted by the insight.
    SeverityValueValuesEnum: Insight's severity.

  Messages:
    ContentValue: A struct of custom fields to explain the insight. Example:
      "grantedPermissionsCount": "1000"

  Fields:
    associatedRecommendations: Recommendations derived from this insight.
    category: Category being targeted by the insight.
    content: A struct of custom fields to explain the insight. Example:
      "grantedPermissionsCount": "1000"
    description: Free-form human readable summary in English. The maximum
      length is 500 characters.
    etag: Fingerprint of the Insight. Provides optimistic locking when
      updating states.
    insightSubtype: Insight subtype. Insight content schema will be stable for
      a given subtype.
    lastRefreshTime: Timestamp of the latest data used to generate the
      insight.
    name: Identifier. Name of the insight.
    observationPeriod: Observation period that led to the insight. The source
      data used to generate the insight ends at last_refresh_time and begins
      at (last_refresh_time - observation_period).
    severity: Insight's severity.
    stateInfo: Information state and metadata.
    targetResources: Fully qualified resource names that this insight is
      targeting.
  """

    class CategoryValueValuesEnum(_messages.Enum):
        """Category being targeted by the insight.

    Values:
      CATEGORY_UNSPECIFIED: Unspecified category.
      COST: The insight is related to cost.
      SECURITY: The insight is related to security.
      PERFORMANCE: The insight is related to performance.
      MANAGEABILITY: This insight is related to manageability.
      SUSTAINABILITY: The insight is related to sustainability.
      RELIABILITY: The insight is related to reliability.
    """
        CATEGORY_UNSPECIFIED = 0
        COST = 1
        SECURITY = 2
        PERFORMANCE = 3
        MANAGEABILITY = 4
        SUSTAINABILITY = 5
        RELIABILITY = 6

    class SeverityValueValuesEnum(_messages.Enum):
        """Insight's severity.

    Values:
      SEVERITY_UNSPECIFIED: Insight has unspecified severity.
      LOW: Insight has low severity.
      MEDIUM: Insight has medium severity.
      HIGH: Insight has high severity.
      CRITICAL: Insight has critical severity.
    """
        SEVERITY_UNSPECIFIED = 0
        LOW = 1
        MEDIUM = 2
        HIGH = 3
        CRITICAL = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ContentValue(_messages.Message):
        """A struct of custom fields to explain the insight. Example:
    "grantedPermissionsCount": "1000"

    Messages:
      AdditionalProperty: An additional property for a ContentValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ContentValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    associatedRecommendations = _messages.MessageField('GoogleCloudRecommenderV1beta1InsightRecommendationReference', 1, repeated=True)
    category = _messages.EnumField('CategoryValueValuesEnum', 2)
    content = _messages.MessageField('ContentValue', 3)
    description = _messages.StringField(4)
    etag = _messages.StringField(5)
    insightSubtype = _messages.StringField(6)
    lastRefreshTime = _messages.StringField(7)
    name = _messages.StringField(8)
    observationPeriod = _messages.StringField(9)
    severity = _messages.EnumField('SeverityValueValuesEnum', 10)
    stateInfo = _messages.MessageField('GoogleCloudRecommenderV1beta1InsightStateInfo', 11)
    targetResources = _messages.StringField(12, repeated=True)