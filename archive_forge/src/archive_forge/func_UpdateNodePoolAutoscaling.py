from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def UpdateNodePoolAutoscaling(self, node_pool_ref, options):
    """Update node pool's autoscaling configuration.

    Args:
      node_pool_ref: node pool Resource to update.
      options: node pool update options

    Returns:
      Updated autoscaling configuration for the node pool.
    """
    pool = self.GetNodePool(node_pool_ref)
    autoscaling = pool.autoscaling
    if autoscaling is None:
        autoscaling = self.messages.NodePoolAutoscaling()
    if options.enable_autoscaling is not None:
        autoscaling.enabled = options.enable_autoscaling
        if not autoscaling.enabled:
            autoscaling.minNodeCount = 0
            autoscaling.maxNodeCount = 0
            autoscaling.totalMinNodeCount = 0
            autoscaling.totalMaxNodeCount = 0
            autoscaling.autoprovisioned = False
            autoscaling.locationPolicy = self.messages.NodePoolAutoscaling.LocationPolicyValueValuesEnum.LOCATION_POLICY_UNSPECIFIED
    if options.enable_autoprovisioning is not None:
        autoscaling.autoprovisioned = options.enable_autoprovisioning
        if autoscaling.autoprovisioned:
            autoscaling.minNodeCount = 0
            autoscaling.totalMinNodeCount = 0
    if options.max_nodes is not None:
        autoscaling.maxNodeCount = options.max_nodes
    if options.min_nodes is not None:
        autoscaling.minNodeCount = options.min_nodes
    if options.total_max_nodes is not None:
        autoscaling.totalMaxNodeCount = options.total_max_nodes
    if options.total_min_nodes is not None:
        autoscaling.totalMinNodeCount = options.total_min_nodes
    if options.location_policy is not None:
        autoscaling.locationPolicy = LocationPolicyEnumFromString(self.messages, options.location_policy)
    return autoscaling