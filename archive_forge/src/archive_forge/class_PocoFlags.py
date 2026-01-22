from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from os import path
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet.policycontroller import protos
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.command_lib.export import util
from googlecloudsdk.core.console import console_io
class PocoFlags:
    """Handle common flags for Poco Commands.

  Use this class to keep command flags that touch similar configuration options
  on the Policy Controller feature in sync across commands.
  """

    def __init__(self, parser: parser_arguments.ArgumentInterceptor, command: str):
        """Constructor.

    Args:
      parser: The argparse parser to add flags to.
      command: The command using this flag utility. i.e. 'enable'.
    """
        self._parser = parser
        self._display_name = command

    @property
    def parser(self):
        return self._parser

    @property
    def display_name(self):
        return self._display_name

    def add_audit_interval(self):
        """Adds handling for audit interval configuration changes."""
        self.parser.add_argument('--audit-interval', type=int, help='How often Policy Controller will audit resources, in seconds.')

    def add_constraint_violation_limit(self):
        """Adds handling for constraint violation limit configuration changes."""
        self.parser.add_argument('--constraint-violation-limit', type=int, help='The number of violations stored on the constraint resource. Must be greater than 0.')

    def add_exemptable_namespaces(self):
        """Adds handling for configuring exemptable namespaces."""
        group = self.parser.add_argument_group('Exemptable Namespace flags.', mutex=True)
        group.add_argument('--exemptable-namespaces', type=str, help='Namespaces that Policy Controller should ignore, separated by commas if multiple are supplied.')
        group.add_argument('--clear-exemptable-namespaces', action='store_true', help='Removes any namespace exemptions, enabling Policy Controller on all namespaces. Setting this flag will overwrite currently exempted namespaces, not append.')

    def add_log_denies_enabled(self):
        """Adds handling for log denies enablement."""
        group = self.parser.add_group('Log Denies flags.', mutex=True)
        group.add_argument('--no-log-denies', action='store_true', help='If set, disable all log denies.')
        group.add_argument('--log-denies', action='store_true', help='If set, log all denies and dry run failures. (To disable, use --no-log-denies)')

    def add_memberships(self):
        """Adds handling for single, multiple or all memberships."""
        group = self.parser.add_argument_group('Membership flags.', mutex=True)
        resources.AddMembershipResourceArg(group, plural=True, membership_help='The membership names to act on, separated by commas if multiple are supplied. Ignored if --all-memberships is supplied; if neither is supplied, a prompt will appear with all available memberships.')
        group.add_argument('--all-memberships', action='store_true', help='If supplied, apply to all Policy Controllers memberships in the fleet.', default=False)

    def add_monitoring(self):
        """Adds handling for monitoring configuration changes."""
        group = self.parser.add_argument_group('Monitoring flags.', mutex=True)
        group.add_argument('--monitoring', type=str, help='Monitoring backend options Policy Controller should export metrics to, separated by commas if multiple are supplied.  Setting this flag will overwrite currently enabled backends, not append. Options: {}'.format(', '.join(constants.MONITORING_BACKENDS)))
        group.add_argument('--no-monitoring', action='store_true', help='Include this flag to disable the monitoring configuration of Policy Controller.')

    def add_mutation(self):
        """Adds handling for mutation enablement."""
        group = self.parser.add_group('Mutation flags.', mutex=True)
        group.add_argument('--no-mutation', action='store_true', help='Disables mutation support.')
        group.add_argument('--mutation', action='store_true', help='If set, enable support for mutation. (To disable, use --no-mutation)')

    def add_no_default_bundles(self):
        self.parser.add_argument('--no-default-bundles', action='store_true', help='If set, skip installing the default bundle of policy-essentials.')

    def add_referential_rules(self):
        """Adds handling for referential rules enablement."""
        group = self.parser.add_group('Referential Rules flags.', mutex=True)
        group.add_argument('--no-referential-rules', action='store_true', help='Disables referential rules support.')
        group.add_argument('--referential-rules', action='store_true', help='If set, enable support for referential constraints. (To disable, use --no-referential-rules)')

    def add_version(self):
        """Adds handling for version flag."""
        self.parser.add_argument('--version', type=str, help='The version of Policy Controller to install; defaults to latest version.')