from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import text
def _NodePoolFromCluster(cluster, node_pool_name):
    """Helper function to get node pool from a cluster, given its name."""
    for node_pool in cluster.nodePools:
        if node_pool.name == node_pool_name:
            return node_pool
    raise NodePoolError('No node pool found matching the name [{}].'.format(node_pool_name))