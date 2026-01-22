from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def ClustersAttributeConfig(cluster_fallthrough=False, global_fallthrough=False):
    """Create a cluster attribute in resource argument.

  Args:
    cluster_fallthrough: If set, enables fallthroughs for the cluster attribute.
    global_fallthrough: If set, enables global fallthroughs for the cluster
      attribute.

  Returns:
    Cluster resource argument parameter config
  """
    fallthroughs = []
    if cluster_fallthrough:
        fallthroughs.append(deps.PropertyFallthrough(properties.VALUES.workstations.cluster))
    if global_fallthrough:
        fallthroughs.append(deps.Fallthrough(lambda: '-', hint='default is all clusters'))
    return concepts.ResourceParameterAttributeConfig(name='cluster', fallthroughs=fallthroughs, help_text='The cluster for the {resource}.')