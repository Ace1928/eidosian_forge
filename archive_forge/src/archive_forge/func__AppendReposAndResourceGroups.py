from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import fnmatch
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
def _AppendReposAndResourceGroups(membership, repos_cross_clusters, cluster_type, name, namespace, source):
    """List all the RepoSync,RootSync CRs and ResourceGroup CRs from the given cluster.

  Args:
    membership: The membership name or cluster name of the current cluster.
    repos_cross_clusters: The repos across multiple clusters.
    cluster_type: The type of the current cluster. It is either a Fleet-cluster
      or a Config-controller cluster.
    name: The name of the desired repo.
    namespace: The namespace of the desired repo.
    source: The source of the repo. It should be copied from the output of the
      list command.

  Returns:
    None

  Raises:
    Error: errors that happen when listing the CRs from the cluster.
  """
    utils.GetConfigManagement(membership)
    params = []
    if not namespace:
        params = ['--all-namespaces']
    else:
        params = ['-n', namespace]
    repos, err = utils.RunKubectl(['get', 'rootsync,reposync,resourcegroup', '-o', 'json'] + params)
    if err:
        raise exceptions.ConfigSyncError('Error getting RootSync,RepoSync,Resourcegroup custom resources: {}'.format(err))
    if not repos:
        return
    obj = json.loads(repos)
    if 'items' not in obj or not obj['items']:
        return
    repos = {}
    resourcegroups = {}
    for item in obj['items']:
        ns, nm = utils.GetObjectKey(item)
        if name and nm != name:
            continue
        key = ns + '/' + nm
        kind = item['kind']
        if kind == 'ResourceGroup':
            resourcegroups[key] = item
        else:
            repos[key] = item
    count = 0
    for key, repo in repos.items():
        repo_source = _GetSourceKey(repo)
        if source and repo_source != source:
            continue
        rg = None
        if key in resourcegroups:
            rg = resourcegroups[key]
        repos_cross_clusters.AddRepo(membership, repo, rg, cluster_type)
        count += 1
    if count > 0:
        log.status.Print('getting {} RepoSync and RootSync from {}'.format(count, membership))