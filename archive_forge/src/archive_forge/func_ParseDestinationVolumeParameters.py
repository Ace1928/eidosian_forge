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
def ParseDestinationVolumeParameters(self, replication, destination_volume_parameters):
    """Parses Destination Volume Parameters for Replication into a config.

    Args:
      replication: The Cloud Netapp Volumes Replication object.
      destination_volume_parameters: The Destination Volume Parameters message
        object.

    Returns:
      Replication message populated with Destination Volume Parameters values.
    """
    if not destination_volume_parameters:
        return
    parameters = self.messages.DestinationVolumeParameters()
    for key, val in destination_volume_parameters.items():
        if key == 'storage_pool':
            parameters.storagePool = val
        elif key == 'volume_id':
            parameters.volumeId = val
        elif key == 'share_name':
            parameters.shareName = val
        elif key == 'description':
            parameters.description = val
        else:
            log.warning('The attribute {} is not recognized.'.format(key))
    replication.destinationVolumeParameters = parameters