from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelDeploymentMonitoringScheduleConfig(_messages.Message):
    """The config for scheduling monitoring job.

  Fields:
    monitorInterval: Required. The model monitoring job scheduling interval.
      It will be rounded up to next full hour. This defines how often the
      monitoring jobs are triggered.
    monitorWindow: The time window of the prediction data being included in
      each prediction dataset. This window specifies how long the data should
      be collected from historical model results for each run. If not set,
      ModelDeploymentMonitoringScheduleConfig.monitor_interval will be used.
      e.g. If currently the cutoff time is 2022-01-08 14:30:00 and the
      monitor_window is set to be 3600, then data from 2022-01-08 13:30:00 to
      2022-01-08 14:30:00 will be retrieved and aggregated to calculate the
      monitoring statistics.
  """
    monitorInterval = _messages.StringField(1)
    monitorWindow = _messages.StringField(2)