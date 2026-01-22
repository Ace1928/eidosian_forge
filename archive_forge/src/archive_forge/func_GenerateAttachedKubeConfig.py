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
def GenerateAttachedKubeConfig(cluster, cluster_id, context, cmd_path):
    """Generates a kubeconfig entry for an Anthos Multi-cloud attached cluster.

  Args:
    cluster: object, Anthos Multi-cloud cluster.
    cluster_id: str, the cluster ID.
    context: str, context for the kubeconfig entry.
    cmd_path: str, authentication provider command path.
  """
    kubeconfig = kubeconfig_util.Kubeconfig.Default()
    kubeconfig.contexts[context] = kubeconfig_util.Context(context, context, context)
    _CheckPreqs()
    _ConnectGatewayKubeconfig(kubeconfig, cluster, cluster_id, context, cmd_path)
    kubeconfig.SetCurrentContext(context)
    kubeconfig.SaveToFile()
    log.status.Print('A new kubeconfig entry "{}" has been generated and set as the current context.'.format(context))