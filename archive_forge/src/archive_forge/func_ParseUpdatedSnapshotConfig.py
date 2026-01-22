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
def ParseUpdatedSnapshotConfig(self, snapshot_config, description=None, labels=None):
    """Parse update information into an updated Snapshot message."""
    if description is not None:
        snapshot_config.description = description
    if labels is not None:
        snapshot_config.labels = labels
    return snapshot_config