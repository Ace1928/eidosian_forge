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
def ParseSnapshotConfig(self, name=None, description=None, labels=None):
    """Parses the command line arguments for Create Snapshot into a config.

    Args:
      name: the name of the Snapshot.
      description: the description of the Snapshot.
      labels: the parsed labels value.

    Returns:
      the configuration that will be used as the request body for creating a
      Cloud NetApp Files Snapshot.
    """
    snapshot = self.messages.Snapshot()
    snapshot.name = name
    snapshot.description = description
    snapshot.labels = labels
    return snapshot