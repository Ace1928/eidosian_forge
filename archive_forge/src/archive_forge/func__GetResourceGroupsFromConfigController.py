from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
def _GetResourceGroupsFromConfigController(project, name, namespace, repo_cluster):
    """List all ResourceGroup CRs from Config Controller clusters.

  Args:
    project: The project id the repo is from.
    name: The name of the corresponding ResourceGroup CR.
    namespace: The namespace of the corresponding ResourceGroup CR.
    repo_cluster: The cluster that the repo is synced to.

  Returns:
    List of raw ResourceGroup dicts

  """
    clusters = []
    resource_groups = []
    try:
        clusters = utils.ListConfigControllerClusters(project)
    except exceptions.ConfigSyncError as err:
        log.error(err)
    if clusters:
        for cluster in clusters:
            if repo_cluster and repo_cluster != cluster[0]:
                continue
            try:
                utils.KubeconfigForCluster(project, cluster[1], cluster[0])
                cc_rg = _GetResourceGroups(cluster[0], name, namespace)
                if cc_rg:
                    resource_groups.extend(cc_rg)
            except exceptions.ConfigSyncError as err:
                log.error(err)
    return resource_groups