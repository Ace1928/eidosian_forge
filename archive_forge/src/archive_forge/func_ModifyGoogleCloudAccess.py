from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def ModifyGoogleCloudAccess(self, cluster_ref, existing_authorized_networks, goole_cloud_access):
    """Update enable_google_cloud_access and schedule cluster update request."""
    authorized_networks = self.messages.MasterAuthorizedNetworksConfig(enabled=existing_authorized_networks.enabled, cidrBlocks=existing_authorized_networks.cidrBlocks, gcpPublicCidrsAccessEnabled=goole_cloud_access)
    update = self.messages.ClusterUpdate(desiredMasterAuthorizedNetworksConfig=authorized_networks)
    op = self.client.projects_locations_clusters.Update(self.messages.UpdateClusterRequest(name=ProjectLocationCluster(cluster_ref.projectId, cluster_ref.zone, cluster_ref.clusterId), update=update))
    return self.ParseOperation(op.name, cluster_ref.zone)