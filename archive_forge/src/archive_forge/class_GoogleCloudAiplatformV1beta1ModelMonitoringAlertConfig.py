from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelMonitoringAlertConfig(_messages.Message):
    """The alert config for model monitoring.

  Fields:
    emailAlertConfig: Email alert config.
    enableLogging: Dump the anomalies to Cloud Logging. The anomalies will be
      put to json payload encoded from proto
      google.cloud.aiplatform.logging.ModelMonitoringAnomaliesLogEntry. This
      can be further sinked to Pub/Sub or any other services supported by
      Cloud Logging.
    notificationChannels: Resource names of the NotificationChannels to send
      alert. Must be of the format `projects//notificationChannels/`
  """
    emailAlertConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelMonitoringAlertConfigEmailAlertConfig', 1)
    enableLogging = _messages.BooleanField(2)
    notificationChannels = _messages.StringField(3, repeated=True)