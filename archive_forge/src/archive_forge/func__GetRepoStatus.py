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
def _GetRepoStatus(rs, git):
    """Get the status for a repo.

  Args:
    rs: The dictionary of a unique repo across multiple clusters.
        It contains the following data: {
           cluster-name-1: RepoSourceGroupPair,
           cluster-name-2: RepoSourceGroupPair }
    git: The string that represent the git spec of the repo.

  Returns:
    One RepoStatus that represents the aggregated
    status for the current repo.
  """
    repo_status = RepoStatus()
    repo_status.source = git
    for _, pair in rs.items():
        status = 'SYNCED'
        obj = pair.repo
        namespace, name = utils.GetObjectKey(obj)
        repo_status.namespace = namespace
        repo_status.name = name
        single_repo_status = _GetStatusForRepo(obj)
        status = single_repo_status.status
        if status == 'SYNCED':
            repo_status.synced += 1
        elif status == 'PENDING':
            repo_status.pending += 1
        elif status == 'ERROR':
            repo_status.error += 1
        elif status == 'STALLED':
            repo_status.stalled += 1
        elif status == 'RECONCILING':
            repo_status.reconciling += 1
        repo_status.total += 1
        repo_status.cluster_type = pair.cluster_type
    return repo_status