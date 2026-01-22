from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class SoleTenancyNodeTemplates(base.Group):
    """Read and manage Compute Engine sole-tenancy node templates.

  Node templates are used to create the nodes in node groups. Nodes are
  Compute Engine servers that are dedicated to your workload. Node templates
  define either the node type or the requirements for a node in terms of
  vCPU, memory, and localSSD.
  """