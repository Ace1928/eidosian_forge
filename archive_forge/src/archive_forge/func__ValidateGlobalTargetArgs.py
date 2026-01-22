from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.forwarding_rules import exceptions
from googlecloudsdk.command_lib.compute.forwarding_rules import flags
from googlecloudsdk.core import properties
def _ValidateGlobalTargetArgs(args):
    """Validate the global forwarding rules args."""
    if args.target_instance:
        raise exceptions.ArgumentError('You cannot specify [--target-instance] for a global forwarding rule.')
    if args.target_pool:
        raise exceptions.ArgumentError('You cannot specify [--target-pool] for a global forwarding rule.')
    if getattr(args, 'backend_service', None):
        raise exceptions.ArgumentError('You cannot specify [--backend-service] for a global forwarding rule.')
    if getattr(args, 'target_vpn_gateway', None):
        raise exceptions.ArgumentError('You cannot specify [--target-vpn-gateway] for a global forwarding rule.')