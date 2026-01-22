from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import gkehub_api_adapter
from googlecloudsdk.api_lib.container.fleet import gkehub_api_util
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.command_lib.projects import util as p_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def DeleteConnectNamespace(kube_client, args):
    """Delete the namespace in the cluster that contains the connect agent.

  Args:
    kube_client: A Kubernetes Client for the cluster to be registered.
    args: an argparse namespace. All arguments that were provided to this
      command invocation.

  Raises:
    calliope_exceptions.MinimumArgumentException: if a kubeconfig file cannot
      be deduced from the command line flags or environment
  """
    namespaces = _GKEConnectNamespace(kube_client, properties.VALUES.core.project.GetOrFail())
    if len(namespaces) > 1:
        log.warning('gcloud will not remove any namespaces containing the Connect Agent since it was found running in multiple namespaces on cluster: [{}]. Please delete these namespaces [{}] maually in your cluster'.format(args.MEMBERSHIP_NAME, namespaces))
        return
    namespace = namespaces[0]
    cleanup_msg = 'Please delete namespace [{}] manually in your cluster.'.format(namespace)
    try:
        kube_util.DeleteNamespace(kube_client, namespace)
    except exceptions.Error:
        log.warning(cleanup_msg)