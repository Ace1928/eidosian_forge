from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def MakeNodeTypeArg():
    return compute_flags.ResourceArgument(resource_name='node types', completer=NodeTypesCompleter, zonal_collection='compute.nodeTypes', zone_explanation=compute_flags.ZONE_PROPERTY_EXPLANATION)