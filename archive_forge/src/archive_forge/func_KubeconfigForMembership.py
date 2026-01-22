from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import fnmatch
import io
import json
import os
import re
import signal
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def KubeconfigForMembership(project, membership):
    """Get the kubeconfig of a membership.

  If the kubeconfig for the membership already exists locally, use it;
  Otherwise run a gcloud command to get the credential for it.

  Args:
    project: The project ID of the membership.
    membership: The name of the membership.

  Returns:
    None

  Raises:
      Error: The error occured when it failed to get credential for the
      membership.
  """
    context = 'connectgateway_{project}_{membership}'.format(project=project, membership=membership)
    command = ['config', 'use-context', context]
    _, err = RunKubectl(command)
    if err is None:
        return
    args = ['container', 'fleet', 'memberships', 'describe', membership, '--project', project, '--format', 'json']
    output, err = _RunGcloud(args)
    if err:
        raise exceptions.ConfigSyncError('Error describing the membership {}: {}'.format(membership, err))
    if output:
        description = json.loads(output)
        cluster_link = description.get('endpoint', {}).get('gkeCluster', {}).get('resourceLink', '')
        if cluster_link:
            m = re.compile('.*/projects/(.*)/locations/(.*)/clusters/(.*)').match(cluster_link)
            project = ''
            location = ''
            cluster = ''
            try:
                project = m.group(1)
                location = m.group(2)
                cluster = m.group(3)
            except IndexError:
                pass
            if project and location and cluster:
                KubeconfigForCluster(project, location, cluster)
                return
    args = ['container', 'fleet', 'memberships', 'get-credentials', membership, '--project', project]
    _, err = _RunGcloud(args)
    if err:
        raise exceptions.ConfigSyncError('Error getting credential for membership {}: {}'.format(membership, err))