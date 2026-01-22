from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1alpha2MarkInsightDismissedRequest(_messages.Message):
    """Request for the `MarkInsightDismissed` method.

  Enums:
    RecommendationChangeTypeValueValuesEnum: Determines whether dismiss state
      will apply to associated recommendations.

  Fields:
    etag: Fingerprint of the Insight. Provides optimistic locking.
    recommendationChangeType: Determines whether dismiss state will apply to
      associated recommendations.
  """

    class RecommendationChangeTypeValueValuesEnum(_messages.Enum):
        """Determines whether dismiss state will apply to associated
    recommendations.

    Values:
      RECOMMENDATION_CHANGE_TYPE_UNSPECIFIED: Unspecified change type. Default
        behavior is DISMISS_RECOMMENDATIONS.
      DISMISS_RECOMMENDATIONS: Dismisses associated recommendations, if any.
        Changes to associated recommendations requires recommender.*.update
        permissions for the linked recommendation types, if applicable.
      LEAVE_RECOMMENDATIONS_UNCHANGED: Makes no changes to the associated
        recommendations.
    """
        RECOMMENDATION_CHANGE_TYPE_UNSPECIFIED = 0
        DISMISS_RECOMMENDATIONS = 1
        LEAVE_RECOMMENDATIONS_UNCHANGED = 2
    etag = _messages.StringField(1)
    recommendationChangeType = _messages.EnumField('RecommendationChangeTypeValueValuesEnum', 2)