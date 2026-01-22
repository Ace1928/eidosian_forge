from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import io
import ipaddress
import os
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
@contextlib.contextmanager
def TemporaryKubeconfig(location_id, cluster_id):
    """Context manager that manages a temporary kubeconfig file for a GKE cluster.

  The kubeconfig file will be automatically created and destroyed and will
  contain only the credentials for the specified GKE cluster. The 'KUBECONFIG'
  value in `os.environ` will be temporarily updated with the temporary
  kubeconfig's path. Consequently, subprocesses started with
  googlecloudsdk.core.execution_utils.Exec while in this context manager will
  see the temporary KUBECONFIG environment variable.

  Args:
    location_id: string, the id of the location to which the cluster belongs
    cluster_id: string, the id of the cluster

  Raises:
    Error: If unable to get credentials for kubernetes cluster.

  Returns:
    the path to the temporary kubeconfig file

  Yields:
    Due to b/73533917, linter crashes without yields.
  """
    gke_util.CheckKubectlInstalled()
    with files.TemporaryDirectory() as tempdir:
        kubeconfig = os.path.join(tempdir, 'kubeconfig')
        old_kubeconfig = encoding.GetEncodedValue(os.environ, KUBECONFIG_ENV_VAR_NAME)
        try:
            encoding.SetEncodedValue(os.environ, KUBECONFIG_ENV_VAR_NAME, kubeconfig)
            gke_api = gke_api_adapter.NewAPIAdapter(GKE_API_VERSION)
            cluster_ref = gke_api.ParseCluster(cluster_id, location_id)
            cluster = gke_api.GetCluster(cluster_ref)
            auth = cluster.masterAuth
            missing_creds = not (auth and auth.clientCertificate and auth.clientKey)
            if missing_creds and (not gke_util.ClusterConfig.UseGCPAuthProvider()):
                raise Error('Unable to get cluster credentials. User must have edit permission on {}'.format(cluster_ref.projectId))
            gke_util.ClusterConfig.Persist(cluster, cluster_ref.projectId)
            yield kubeconfig
        finally:
            encoding.SetEncodedValue(os.environ, KUBECONFIG_ENV_VAR_NAME, old_kubeconfig)