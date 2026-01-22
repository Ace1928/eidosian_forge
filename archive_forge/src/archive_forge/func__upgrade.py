from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.container.gkeonprem import operations
from googlecloudsdk.api_lib.container.gkeonprem import vmware_admin_clusters
from googlecloudsdk.api_lib.container.gkeonprem import vmware_clusters
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.command_lib.container.gkeonprem import flags as common_flags
from googlecloudsdk.command_lib.container.vmware import constants
from googlecloudsdk.command_lib.container.vmware import errors
from googlecloudsdk.command_lib.container.vmware import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.util import semver
def _upgrade(self, args, cluster_ref):
    log.status.Print('Upgrading Anthos on VMware user cluster [{}]'.format(cluster_ref))
    cluster_client = vmware_clusters.ClustersClient()
    operation_client = operations.OperationsClient()
    operation = cluster_client.Update(args)
    operation_response = operation_client.Wait(operation)
    log.UpdatedResource(cluster_ref, 'Anthos on VMware user cluster')
    return operation_response