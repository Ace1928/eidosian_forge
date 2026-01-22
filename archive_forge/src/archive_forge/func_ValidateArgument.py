from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import firewalls_utils
from googlecloudsdk.api_lib.compute import utils as compute_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute.firewall_rules import flags
def ValidateArgument(self, messages, args):
    self.new_allowed = firewalls_utils.ParseRules(args.allow, messages, firewalls_utils.ActionType.ALLOW)
    args_unset = all((x is None for x in (args.allow, args.description, args.source_ranges, args.source_tags, args.target_tags)))
    if self.with_egress_firewall:
        args_unset = args_unset and all((x is None for x in (args.destination_ranges, args.priority, args.rules)))
    if self.with_service_account:
        args_unset = args_unset and all((x is None for x in (args.source_service_accounts, args.target_service_accounts)))
    args_unset = args_unset and args.disabled is None
    args_unset = args_unset and args.enable_logging is None
    args_unset = args_unset and (not args.logging_metadata)
    if args_unset:
        raise exceptions.UpdatePropertyError('At least one property must be modified.')
    if args.rules and args.allow:
        raise firewalls_utils.ArgumentValidationError('Can NOT specify --rules and --allow in the same request.')