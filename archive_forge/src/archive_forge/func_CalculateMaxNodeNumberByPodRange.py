from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
def CalculateMaxNodeNumberByPodRange(cluster_ipv4_cidr):
    """Calculate the maximum number of nodes for route based clusters.

  Args:
    cluster_ipv4_cidr: The cluster IPv4 CIDR requested. If cluster_ipv4_cidr is
      not specified, GKE_DEFAULT_POD_RANGE will be used.

  Returns:
    The maximum number of nodes the cluster can have.
    The function returns -1 in case of error.
  """
    if cluster_ipv4_cidr is None:
        pod_range = GKE_DEFAULT_POD_RANGE
    else:
        blocksize = cluster_ipv4_cidr.split('/')[-1]
        if not blocksize.isdecimal():
            return -1
        pod_range = int(blocksize)
        if pod_range < 0:
            return -1
    pod_range_ips = 2 ** (32 - pod_range) - 2 ** (32 - GKE_ROUTE_BASED_SERVICE_RANGE)
    pod_range_ips_per_node = 2 ** (32 - GKE_DEFAULT_POD_RANGE_PER_NODE)
    if pod_range_ips < pod_range_ips_per_node:
        return -1
    return int(pod_range_ips / pod_range_ips_per_node)