from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.routers.nats.rules import flags
from googlecloudsdk.core import exceptions as core_exceptions
import six
def CreateRuleMessage(args, compute_holder, nat):
    """Creates a Rule message from the specified arguments."""
    rule = compute_holder.client.messages.RouterNatRule(ruleNumber=args.rule_number, match=args.match, action=compute_holder.client.messages.RouterNatRuleAction())
    is_private_nat = _IsPrivateNat(nat, compute_holder)
    if args.source_nat_active_ips:
        if is_private_nat:
            raise ActiveIpsNotSupportedError()
        rule.action.sourceNatActiveIps = [six.text_type(ip) for ip in flags.ACTIVE_IPS_ARG_REQUIRED.ResolveAsResource(args, compute_holder.resources)]
    elif not is_private_nat:
        raise ActiveIpsRequiredError()
    if args.source_nat_active_ranges:
        if not is_private_nat:
            raise ActiveRangesNotSupportedError()
        rule.action.sourceNatActiveRanges = [six.text_type(subnet) for subnet in flags.ACTIVE_RANGES_ARG.ResolveAsResource(args, compute_holder.resources)]
    elif is_private_nat:
        raise ActiveRangesRequiredError()
    return rule