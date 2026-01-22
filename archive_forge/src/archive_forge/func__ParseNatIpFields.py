from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.networks.subnets import flags as subnet_flags
from googlecloudsdk.command_lib.compute.routers.nats import flags as nat_flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def _ParseNatIpFields(args, compute_holder):
    messages = compute_holder.client.messages
    if args.auto_allocate_nat_external_ips:
        return (messages.RouterNat.NatIpAllocateOptionValueValuesEnum.AUTO_ONLY, list())
    return (messages.RouterNat.NatIpAllocateOptionValueValuesEnum.MANUAL_ONLY, [six.text_type(address) for address in nat_flags.IP_ADDRESSES_ARG.ResolveAsResource(args, compute_holder.resources)])