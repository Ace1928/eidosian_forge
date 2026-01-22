from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.security_policies import flags as security_policy_flags
from googlecloudsdk.command_lib.compute.security_policies.rules import flags
from googlecloudsdk.core import properties
@classmethod
def _UpdatePreconfigWafConfig(cls, compute_client, existing_rule, args):
    """Updates Preconfig WafConfig."""
    new_preconfig_waf_config = compute_client.messages.SecurityPolicyRulePreconfiguredWafConfig()
    if args.target_rule_set == '*':
        return new_preconfig_waf_config
    has_request_field_args = False
    if args.IsSpecified('request_header_to_exclude') or args.IsSpecified('request_cookie_to_exclude') or args.IsSpecified('request_query_param_to_exclude') or args.IsSpecified('request_uri_to_exclude'):
        has_request_field_args = True
    if existing_rule.preconfiguredWafConfig:
        exclusions = existing_rule.preconfiguredWafConfig.exclusions
    else:
        exclusions = []
    for exclusion in exclusions:
        if cls._IsIdenticalTarget(exclusion, args.target_rule_set, args.target_rule_ids or []):
            if has_request_field_args:
                new_exclusion = cls._UpdateExclusion(compute_client, exclusion, args.request_header_to_exclude, args.request_cookie_to_exclude, args.request_query_param_to_exclude, args.request_uri_to_exclude)
                if new_exclusion:
                    new_preconfig_waf_config.exclusions.append(new_exclusion)
        else:
            new_preconfig_waf_config.exclusions.append(exclusion)
    return new_preconfig_waf_config