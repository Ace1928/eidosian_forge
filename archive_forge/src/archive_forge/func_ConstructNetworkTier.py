from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import name_generator
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.addresses import flags
import ipaddr
from six.moves import zip  # pylint: disable=redefined-builtin
def ConstructNetworkTier(self, messages, args):
    if args.network_tier:
        network_tier = args.network_tier.upper()
        if network_tier in constants.NETWORK_TIER_CHOICES_FOR_INSTANCE:
            return messages.Address.NetworkTierValueValuesEnum(args.network_tier)
        else:
            raise exceptions.InvalidArgumentException('--network-tier', 'Invalid network tier [{tier}]'.format(tier=network_tier))
    else:
        return None