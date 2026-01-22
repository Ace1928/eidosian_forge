from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityFeedbackFeedbackContext(_messages.Message):
    """FeedbackContext captures the intent of the submitted feedback.

  Enums:
    FeedbackTypeValueValuesEnum: Required. The type of feedback being
      submitted.

  Fields:
    attribute: Required. The API attribute the user is providing feedback
      about. Supported values: - useragent - client_received_start_timestamp -
      apiproxy - client_id - organization - environment - request_uri -
      proxy_basepath - ax_resolved_client_ip - request_size - response_size -
      is_error - ax_geo_country - access_token - developer_app - incident_id -
      incident_name - api_product - developer_email - response_status_code -
      bot_reason - target_url
    feedbackType: Required. The type of feedback being submitted.
    value: Required. The value of the attribute the user is providing feedback
      about.
  """

    class FeedbackTypeValueValuesEnum(_messages.Enum):
        """Required. The type of feedback being submitted.

    Values:
      FEEDBACK_TYPE_UNSPECIFIED: Unspecified feedback type.
      DETECTION_FALSE_POSITIVE: Feedback identifying an incorrect
        classification by an ML model.
      DETECTION_FALSE_NEGATIVE: Feedback identifying a classification by an ML
        model that was missed.
    """
        FEEDBACK_TYPE_UNSPECIFIED = 0
        DETECTION_FALSE_POSITIVE = 1
        DETECTION_FALSE_NEGATIVE = 2
    attribute = _messages.StringField(1)
    feedbackType = _messages.EnumField('FeedbackTypeValueValuesEnum', 2)
    value = _messages.StringField(3)