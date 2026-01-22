from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.security_policies import flags
from googlecloudsdk.command_lib.compute.security_policies import security_policies_utils
@base.ReleaseTracks(base.ReleaseTrack.GA)
class UpdateGa(base.UpdateCommand):
    """Update a Compute Engine security policy.

  *{command}* is used to update security policies.

  ## EXAMPLES

  To update the description run this:

    $ {command} SECURITY_POLICY --description='new description'
  """
    SECURITY_POLICY_ARG = None

    @classmethod
    def Args(cls, parser):
        cls.SECURITY_POLICY_ARG = flags.SecurityPolicyMultiScopeArgument()
        cls.SECURITY_POLICY_ARG.AddArgument(parser, operation_type='update')
        parser.add_argument('--description', help='An optional, textual description for the security policy.')
        flags.AddCloudArmorAdaptiveProtection(parser)
        flags.AddAdvancedOptions(parser)
        flags.AddRecaptchaOptions(parser)
        flags.AddDdosProtectionConfig(parser)

    def _ValidateArgs(self, args):
        """Validates that at least one field to update is specified.

    Args:
      args: The arguments given to the update command.
    """
        if not (args.IsSpecified('description') or args.IsSpecified('enable_layer7_ddos_defense') or args.IsSpecified('layer7_ddos_defense_rule_visibility') or args.IsSpecified('json_parsing') or args.IsSpecified('json_custom_content_types') or args.IsSpecified('log_level') or args.IsSpecified('recaptcha_redirect_site_key') or args.IsSpecified('network_ddos_protection') or args.IsSpecified('user_ip_request_headers')):
            parameter_names = ['--description', '--enable-layer7-ddos-defense', '--layer7-ddos-defense-rule-visibility', '--json-parsing', '--json-custom-content-types', '--log-level', '--recaptcha-redirect-site-key', '--network-ddos-protection', '--user-ip-request-headers']
            raise exceptions.MinimumArgumentException(parameter_names, 'Please specify at least one property to update')

    def Run(self, args):
        self._ValidateArgs(args)
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        ref = self.SECURITY_POLICY_ARG.ResolveAsResource(args, holder.resources, default_scope=compute_scope.ScopeEnum.GLOBAL)
        security_policy = client.SecurityPolicy(ref=ref, compute_client=holder.client)
        existing_security_policy = security_policy.Describe()[0]
        description = existing_security_policy.description
        adaptive_protection_config = existing_security_policy.adaptiveProtectionConfig
        advanced_options_config = existing_security_policy.advancedOptionsConfig
        recaptcha_options_config = existing_security_policy.recaptchaOptionsConfig
        ddos_protection_config = existing_security_policy.ddosProtectionConfig
        if args.description is not None:
            description = args.description
        if args.IsSpecified('enable_layer7_ddos_defense') or args.IsSpecified('layer7_ddos_defense_rule_visibility'):
            adaptive_protection_config = security_policies_utils.CreateAdaptiveProtectionConfig(holder.client, args, adaptive_protection_config)
        if args.IsSpecified('json_parsing') or args.IsSpecified('json_custom_content_types') or args.IsSpecified('log_level') or args.IsSpecified('user_ip_request_headers'):
            advanced_options_config = security_policies_utils.CreateAdvancedOptionsConfig(holder.client, args, advanced_options_config)
        if args.IsSpecified('recaptcha_redirect_site_key'):
            recaptcha_options_config = security_policies_utils.CreateRecaptchaOptionsConfig(holder.client, args, recaptcha_options_config)
        if args.IsSpecified('network_ddos_protection'):
            ddos_protection_config = security_policies_utils.CreateDdosProtectionConfig(holder.client, args, ddos_protection_config)
        updated_security_policy = holder.client.messages.SecurityPolicy(description=description, adaptiveProtectionConfig=adaptive_protection_config, advancedOptionsConfig=advanced_options_config, recaptchaOptionsConfig=recaptcha_options_config, ddosProtectionConfig=ddos_protection_config, fingerprint=existing_security_policy.fingerprint)
        return security_policy.Patch(security_policy=updated_security_policy)