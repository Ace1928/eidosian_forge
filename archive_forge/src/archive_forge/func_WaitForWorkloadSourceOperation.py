from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import List, Optional
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.iam import identity_pool_waiter
from googlecloudsdk.core import resources as sdkresources
def WaitForWorkloadSourceOperation(client, lro_ref, for_managed_identity: bool=False, delete: bool=False):
    """Make API calls to poll for a workload source LRO.

  Args:
    client: the iam v1 client.
    lro_ref: the lro ref returned from a LRO workload source API call.
    for_managed_identity: whether the workload source LRO is under a managed
      identity
    delete: whether it's a delete operation

  Returns:
    The result workload source or None for delete
  """
    lro_resource = sdkresources.REGISTRY.ParseRelativeName(lro_ref.name, collection='iam.projects.locations.workloadIdentityPools.namespaces.managedIdentities.workloadSources.operations' if for_managed_identity else 'iam.projects.locations.workloadIdentityPools.namespaces.workloadSources')
    if delete:
        result = waiter.WaitFor(identity_pool_waiter.IdentityPoolOperationPollerNoResources(client.projects_locations_workloadIdentityPools_namespaces_workloadSources, client.projects_locations_workloadIdentityPools_namespaces_workloadSources_operations), lro_resource, 'Waiting for operation [{}] to complete'.format(lro_ref.name), max_wait_ms=300000)
    else:
        result = waiter.WaitFor(identity_pool_waiter.IdentityPoolOperationPoller(client.projects_locations_workloadIdentityPools_namespaces_workloadSources, client.projects_locations_workloadIdentityPools_namespaces_workloadSources_operations), lro_resource, 'Waiting for operation [{}] to complete'.format(lro_ref.name), max_wait_ms=300000)
    return result