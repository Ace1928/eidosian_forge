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
def GenerateKubeconfig(cluster, cluster_id, context, cmd_path, cmd_args, private_ep=False):
    """Generates a kubeconfig entry for an Anthos Multi-cloud cluster.

  Args:
    cluster: object, Anthos Multi-cloud cluster.
    cluster_id: str, the cluster ID.
    context: str, context for the kubeconfig entry.
    cmd_path: str, authentication provider command path.
    cmd_args: str, authentication provider command arguments.
    private_ep: bool, whether to use private VPC for authentication.

  Raises:
      Error: don't have the permission to open kubeconfig file.
  """
    kubeconfig = kubeconfig_util.Kubeconfig.Default()
    kubeconfig.contexts[context] = kubeconfig_util.Context(context, context, context)
    version = _GetSemver(cluster, cluster_id)
    if private_ep or version < semver.SemVer('1.21.0'):
        _CheckPreqs(private_endpoint=True)
        _PrivateVPCKubeconfig(kubeconfig, cluster, cluster_id, context, cmd_path, cmd_args)
    else:
        _CheckPreqs()
        _ConnectGatewayKubeconfig(kubeconfig, cluster, cluster_id, context, cmd_path)
    kubeconfig.SetCurrentContext(context)
    kubeconfig.SaveToFile()
    log.status.Print('A new kubeconfig entry "{}" has been generated and set as the current context.'.format(context))