from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import clusters
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.bigtable import arguments
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def _ValidateClusterConfigArgs(self, cluster_config):
    """Validates arguments in cluster-config as a repeated dict."""
    for cluster_dict in cluster_config:
        if 'autoscaling-min-nodes' in cluster_dict or 'autoscaling-max-nodes' in cluster_dict or 'autoscaling-cpu-target' in cluster_dict or ('autoscaling-storage-target' in cluster_dict):
            if 'nodes' in cluster_dict:
                raise exceptions.InvalidArgumentException('--autoscaling-min-nodes --autoscaling-max-nodes --autoscaling-cpu-target --autoscaling-storage-target', 'At most one of nodes | autoscaling-min-nodes autoscaling-max-nodes autoscaling-cpu-target autoscaling-storage-target may be specified in --cluster-config')
            if 'autoscaling-min-nodes' not in cluster_dict or 'autoscaling-max-nodes' not in cluster_dict or 'autoscaling-cpu-target' not in cluster_dict:
                raise exceptions.InvalidArgumentException('--autoscaling-min-nodes --autoscaling-max-nodes --autoscaling-cpu-target', 'All of --autoscaling-min-nodes --autoscaling-max-nodes --autoscaling-cpu-target must be set to enable Autoscaling.')