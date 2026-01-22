from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
def GetDiscoveredWorkloadRef(args):
    """Returns a discovered workload reference."""
    discovered_workload_ref = args.CONCEPTS.discovered_workload.Parse()
    if not discovered_workload_ref.Name():
        raise exceptions.InvalidArgumentException('discovered workload', 'discovered workload id must be non-empty.')
    return discovered_workload_ref