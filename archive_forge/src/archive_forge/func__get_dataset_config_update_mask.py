from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
def _get_dataset_config_update_mask(self, retention_period=None, description=None):
    """Returns the update_mask list."""
    update_mask = []
    if retention_period is not None:
        update_mask.append('retentionPeriodDays')
    if description is not None:
        update_mask.append('description')
    return update_mask