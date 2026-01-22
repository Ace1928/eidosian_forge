from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import os
import subprocess
from googlecloudsdk.api_lib.container import kubeconfig as kubeconfig_util
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.command_lib.container.fleet import gateway
from googlecloudsdk.command_lib.container.fleet import gwkubeconfig_util
from googlecloudsdk.command_lib.container.gkemulticloud import errors
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
def _PrivateVPCKubeconfig(kubeconfig, cluster, cluster_id, context, cmd_path, cmd_args):
    """Generates the kubeconfig entry to connect using private VPC.

  Args:
    kubeconfig: object, Kubeconfig object.
    cluster: object, Anthos Multi-cloud cluster.
    cluster_id: str, the cluster ID.
    context: str, context for the kubeconfig entry.
    cmd_path: str, authentication provider command path.
    cmd_args: str, authentication provider command arguments.
  """
    user = {}
    user['exec'] = _ExecAuthPlugin(cmd_path, cmd_args)
    kubeconfig.users[context] = {'name': context, 'user': user}
    cluster_kwargs = {}
    if cluster.clusterCaCertificate is None:
        log.warning('Cluster is missing certificate authority data.')
    else:
        cluster_kwargs['ca_data'] = _GetCaData(cluster.clusterCaCertificate)
    if cluster.endpoint is None:
        raise errors.MissingClusterField(cluster_id, 'endpoint', STILL_PROVISIONING_MSG)
    kubeconfig.clusters[context] = kubeconfig_util.Cluster(context, 'https://{}'.format(cluster.endpoint), **cluster_kwargs)