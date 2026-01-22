from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RolloutNotificationEvent(_messages.Message):
    """Payload proto for "clouddeploy.googleapis.com/rollout_notification"
  Platform Log event that describes the failure to send rollout status change
  Pub/Sub notification.

  Enums:
    TypeValueValuesEnum: Type of this notification, e.g. for a Pub/Sub
      failure.

  Fields:
    message: Debug message for when a notification fails to send.
    pipelineUid: Unique identifier of the `DeliveryPipeline`.
    release: The name of the `Release`.
    releaseUid: Unique identifier of the `Release`.
    rollout: The name of the `Rollout`.
    rolloutUid: Unique identifier of the `Rollout`.
    targetId: ID of the `Target` that the rollout is deployed to.
    type: Type of this notification, e.g. for a Pub/Sub failure.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Type of this notification, e.g. for a Pub/Sub failure.

    Values:
      TYPE_UNSPECIFIED: Type is unspecified.
      TYPE_PUBSUB_NOTIFICATION_FAILURE: A Pub/Sub notification failed to be
        sent.
      TYPE_RESOURCE_STATE_CHANGE: Resource state changed.
      TYPE_PROCESS_ABORTED: A process aborted.
      TYPE_RESTRICTION_VIOLATED: Restriction check failed.
      TYPE_RESOURCE_DELETED: Resource deleted.
      TYPE_ROLLOUT_UPDATE: Rollout updated.
      TYPE_DEPLOY_POLICY_EVALUATION: Deploy Policy evaluation.
      TYPE_RENDER_STATUES_CHANGE: Deprecated: This field is never used. Use
        release_render log type instead.
    """
        TYPE_UNSPECIFIED = 0
        TYPE_PUBSUB_NOTIFICATION_FAILURE = 1
        TYPE_RESOURCE_STATE_CHANGE = 2
        TYPE_PROCESS_ABORTED = 3
        TYPE_RESTRICTION_VIOLATED = 4
        TYPE_RESOURCE_DELETED = 5
        TYPE_ROLLOUT_UPDATE = 6
        TYPE_DEPLOY_POLICY_EVALUATION = 7
        TYPE_RENDER_STATUES_CHANGE = 8
    message = _messages.StringField(1)
    pipelineUid = _messages.StringField(2)
    release = _messages.StringField(3)
    releaseUid = _messages.StringField(4)
    rollout = _messages.StringField(5)
    rolloutUid = _messages.StringField(6)
    targetId = _messages.StringField(7)
    type = _messages.EnumField('TypeValueValuesEnum', 8)