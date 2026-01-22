from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
def _GetResourceGroups(cluster_name, name, namespace):
    """List all the ResourceGroup CRs from the given cluster.

  Args:
    cluster_name: The membership name or cluster name of the current cluster.
    name: The name of the desired ResourceGroup.
    namespace: The namespace of the desired ResourceGroup.

  Returns:
    List of raw ResourceGroup dicts

  Raises:
    Error: errors that happen when listing the CRs from the cluster.
  """
    utils.GetConfigManagement(cluster_name)
    if not namespace:
        params = ['--all-namespaces']
    else:
        params = ['-n', namespace]
    repos, err = utils.RunKubectl(['get', 'resourcegroup.kpt.dev', '-o', 'json'] + params)
    if err:
        raise exceptions.ConfigSyncError('Error getting ResourceGroup custom resources for cluster {}: {}'.format(cluster_name, err))
    if not repos:
        return []
    obj = json.loads(repos)
    if 'items' not in obj or not obj['items']:
        return []
    resource_groups = []
    for item in obj['items']:
        _, nm = utils.GetObjectKey(item)
        if name and nm != name:
            continue
        resource_groups.append(RawResourceGroup(cluster_name, item))
    return resource_groups