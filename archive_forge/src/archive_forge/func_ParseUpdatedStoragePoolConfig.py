from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util as netapp_api_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ParseUpdatedStoragePoolConfig(self, storagepool_config, description=None, active_directory=None, labels=None, capacity=None, allow_auto_tiering=None):
    """Parse update information into an updated Storage Pool message."""
    if capacity is not None:
        storagepool_config.capacityGib = capacity
    if active_directory is not None:
        storagepool_config.activeDirectory = active_directory
    if description is not None:
        storagepool_config.description = description
    if allow_auto_tiering is not None:
        storagepool_config.allowAutoTiering = allow_auto_tiering
    if labels is not None:
        storagepool_config.labels = labels
    return storagepool_config