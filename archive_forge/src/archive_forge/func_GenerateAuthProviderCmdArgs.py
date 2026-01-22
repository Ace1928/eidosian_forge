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
def GenerateAuthProviderCmdArgs(kind, cluster_id, location, project):
    """Generates command arguments for kubeconfig's authorization provider.

  Args:
    kind: str, kind of the cluster e.g. aws, azure.
    cluster_id: str, ID of the cluster.
    location: str, Google location of the cluster.
    project: str, Google Cloud project of the cluster.

  Returns:
    The command arguments for kubeconfig's authorization provider.
  """
    template = 'container {kind} clusters print-access-token {cluster_id} --project={project} --location={location} --format=json --exec-credential'
    return template.format(kind=kind, cluster_id=cluster_id, location=location, project=project)