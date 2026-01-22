from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.network_attachments import flags
from googlecloudsdk.command_lib.compute.networks.subnets import flags as subnetwork_flags
def _GetSubnetworks(self, holder, args):
    """Returns subnetwork urls from the argument."""
    subnetwork_refs = self.SUBNETWORK_ARG.ResolveAsResource(args, holder.resources, default_scope=compute_scope.ScopeEnum.REGION, scope_lister=compute_flags.GetDefaultScopeLister(holder.client))
    subnetworks = [subnetwork_ref.SelfLink() for subnetwork_ref in subnetwork_refs]
    return subnetworks