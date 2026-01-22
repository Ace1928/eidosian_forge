from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicyanalyzerV1beta1Activity(_messages.Message):
    """A GoogleCloudPolicyanalyzerV1beta1Activity object.

  Messages:
    ActivityValue: A struct of custom fields to explain the activity.

  Fields:
    activity: A struct of custom fields to explain the activity.
    activityType: The type of the activity.
    fullResourceName: The full resource name that identifies the resource. For
      examples of full resource names for Google Cloud services, see
      https://cloud.google.com/iam/help/troubleshooter/full-resource-names.
    observationPeriod: The data observation period to build the activity.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ActivityValue(_messages.Message):
        """A struct of custom fields to explain the activity.

    Messages:
      AdditionalProperty: An additional property for a ActivityValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ActivityValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    activity = _messages.MessageField('ActivityValue', 1)
    activityType = _messages.StringField(2)
    fullResourceName = _messages.StringField(3)
    observationPeriod = _messages.MessageField('GoogleCloudPolicyanalyzerV1beta1ObservationPeriod', 4)