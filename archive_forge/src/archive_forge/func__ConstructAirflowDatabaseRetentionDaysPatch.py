from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
def _ConstructAirflowDatabaseRetentionDaysPatch(airflow_database_retention_days, release_track):
    """Constructs an environment patch for Airflow Database Retention feature.

  Args:
    airflow_database_retention_days: int or None, the number of retention days
      for airflow database data retention mechanism
    release_track: base.ReleaseTrack, the release track of command. It dictates
      which Composer client library is used.

  Returns:
    (str, Environment), the field mask and environment to use for update.
  """
    messages = api_util.GetMessagesModule(release_track=release_track)
    config = messages.EnvironmentConfig()
    if airflow_database_retention_days == 0:
        config.dataRetentionConfig = messages.DataRetentionConfig(airflowMetadataRetentionConfig=messages.AirflowMetadataRetentionPolicyConfig(retentionMode=messages.AirflowMetadataRetentionPolicyConfig.RetentionModeValueValuesEnum.RETENTION_MODE_DISABLED))
    else:
        config.dataRetentionConfig = messages.DataRetentionConfig(airflowMetadataRetentionConfig=messages.AirflowMetadataRetentionPolicyConfig(retentionDays=airflow_database_retention_days, retentionMode=messages.AirflowMetadataRetentionPolicyConfig.RetentionModeValueValuesEnum.RETENTION_MODE_ENABLED))
    return ('config.data_retention_configuration.airflow_metadata_retention_config', messages.Environment(config=config))