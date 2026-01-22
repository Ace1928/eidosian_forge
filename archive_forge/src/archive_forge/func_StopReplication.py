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
def StopReplication(self, replication_ref, force):
    """Send a stop request for the Cloud NetApp Volume Replication."""
    stop_request = self.messages.NetappProjectsLocationsVolumesReplicationsStopRequest(name=replication_ref.RelativeName(), stopReplicationRequest=self.messages.StopReplicationRequest(force=force))
    return self.client.projects_locations_volumes_replications.Stop(stop_request)