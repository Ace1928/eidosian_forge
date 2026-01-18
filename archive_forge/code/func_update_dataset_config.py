from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
def update_dataset_config(self, dataset_config_relative_name, retention_period=None, description=None):
    """Updates the dataset config.

    Args:
      dataset_config_relative_name (str): The relative name of the dataset
        config to be modified.
      retention_period (int): No of days for which insights data is to be
        retained in BigQuery instance.
      description (str): Human readable description text for the given dataset
        config.

    Returns:
      An instance of Operation message.
    """
    update_mask = self._get_dataset_config_update_mask(retention_period, description)
    if not update_mask:
        raise errors.InsightApiError('Nothing to update for dataset config: {}'.format(dataset_config_relative_name))
    dataset_config = self.messages.DatasetConfig(retentionPeriodDays=retention_period, description=description)
    request = self.messages.StorageinsightsProjectsLocationsDatasetConfigsPatchRequest(name=dataset_config_relative_name, datasetConfig=dataset_config, updateMask=','.join(update_mask))
    return self.client.projects_locations_datasetConfigs.Patch(request)