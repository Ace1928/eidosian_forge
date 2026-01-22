from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
import json
import os
import re
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.container import util as c_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.container.fleet import format_util
from googlecloudsdk.command_lib.container.fleet.memberships import gke_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from kubernetes import client as kube_client_lib
from kubernetes import config as kube_client_config
from six.moves.urllib.parse import urljoin
def ValidateClusterIdentifierFlags(kube_client, args):
    """Validates if --gke-cluster | --gke-uri is supplied for GKE cluster, and --context for non GKE clusters.

  Args:
    kube_client: A Kubernetes client for the cluster to be registered.
    args: An argparse namespace. All arguments that were provided to this
      command invocation.

  Raises:
    calliope_exceptions.ConflictingArgumentsException: --context, --gke-uri,
    --gke-cluster are conflicting arguments.
    calliope_exceptions.ConflictingArgumentsException is raised if more than
    one of these arguments is set.

    calliope_exceptions.InvalidArgumentException is raised if --context is set
    for non GKE clusters.
  """
    is_gke_cluster = IsGKECluster(kube_client)
    if args.context and is_gke_cluster:
        raise calliope_exceptions.InvalidArgumentException('--context', '--context cannot be used for GKE clusters. Either --gke-uri | --gke-cluster must be specified')
    if args.gke_uri and (not is_gke_cluster):
        raise calliope_exceptions.InvalidArgumentException('--gke-uri', 'use --context for non GKE clusters.')
    if args.gke_cluster and (not is_gke_cluster):
        raise calliope_exceptions.InvalidArgumentException('--gke-cluster', 'use --context for non GKE clusters.')