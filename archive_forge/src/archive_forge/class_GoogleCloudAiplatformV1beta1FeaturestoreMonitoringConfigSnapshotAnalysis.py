from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FeaturestoreMonitoringConfigSnapshotAnalysis(_messages.Message):
    """Configuration of the Featurestore's Snapshot Analysis Based Monitoring.
  This type of analysis generates statistics for each Feature based on a
  snapshot of the latest feature value of each entities every
  monitoring_interval.

  Fields:
    disabled: The monitoring schedule for snapshot analysis. For EntityType-
      level config: unset / disabled = true indicates disabled by default for
      Features under it; otherwise by default enable snapshot analysis
      monitoring with monitoring_interval for Features under it. Feature-level
      config: disabled = true indicates disabled regardless of the EntityType-
      level config; unset monitoring_interval indicates going with EntityType-
      level config; otherwise run snapshot analysis monitoring with
      monitoring_interval regardless of the EntityType-level config.
      Explicitly Disable the snapshot analysis based monitoring.
    monitoringInterval: Configuration of the snapshot analysis based
      monitoring pipeline running interval. The value is rolled up to full
      day. If both monitoring_interval_days and the deprecated
      `monitoring_interval` field are set when creating/updating
      EntityTypes/Features, monitoring_interval_days will be used.
    monitoringIntervalDays: Configuration of the snapshot analysis based
      monitoring pipeline running interval. The value indicates number of
      days.
    stalenessDays: Customized export features time window for snapshot
      analysis. Unit is one day. Default value is 3 weeks. Minimum value is 1
      day. Maximum value is 4000 days.
  """
    disabled = _messages.BooleanField(1)
    monitoringInterval = _messages.StringField(2)
    monitoringIntervalDays = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    stalenessDays = _messages.IntegerField(4, variant=_messages.Variant.INT32)