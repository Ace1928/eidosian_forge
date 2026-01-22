from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import forwarding_rules_utils as utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import exceptions as fw_exceptions
from googlecloudsdk.command_lib.compute.forwarding_rules import flags
from googlecloudsdk.core import log
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _ProcessCommonArgs(self, client, resources, args, forwarding_rule_ref, forwarding_rule):
    """Processes common arguments for global and regional commands.

    Args:
      client: The client used by gcloud.
      resources: The resource parser.
      args: The arguments passed to the gcloud command.
      forwarding_rule_ref: The forwarding rule reference.
      forwarding_rule: The forwarding rule to set properties on.
    """
    if args.ip_version:
        forwarding_rule.ipVersion = client.messages.ForwardingRule.IpVersionValueValuesEnum(args.ip_version)
    if args.network:
        forwarding_rule.network = flags.NetworkArg().ResolveAsResource(args, resources).SelfLink()
    if args.subnet:
        if not args.subnet_region and forwarding_rule_ref.Collection() == 'compute.forwardingRules':
            args.subnet_region = forwarding_rule_ref.region
        forwarding_rule.subnetwork = flags.SUBNET_ARG.ResolveAsResource(args, resources).SelfLink()