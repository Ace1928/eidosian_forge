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
def ParseReplicationConfig(self, name=None, description=None, labels=None, replication_schedule=None, destination_volume_parameters=None):
    """Parse the command line arguments for Create Replication into a config.

    Args:
      name: the name of the Replication.
      description: the description of the Replication.
      labels: the parsed labels value.
      replication_schedule: the schedule for Replication.
      destination_volume_parameters: the input parameters used for creating
        destination volume.

    Returns:
      the configuration that will be used as the request body for creating a
      Cloud NetApp Files Replication.
    """
    replication = self.messages.Replication()
    replication.name = name
    replication.description = description
    replication.labels = labels
    replication.replicationSchedule = replication_schedule
    self._adapter.ParseDestinationVolumeParameters(replication, destination_volume_parameters)
    return replication