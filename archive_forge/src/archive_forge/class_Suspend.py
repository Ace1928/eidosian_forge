from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import command
from googlecloudsdk.command_lib.container.fleet.policycontroller import flags
class Suspend(base.UpdateCommand, command.PocoCommand):
    """Suspend Policy Controller Feature.

  Suspends the Policy Controller. This will disable all kubernetes webhooks on
  the configured cluster, thereby removing admission and mutation functionality.
  Audit functionality will remain in place.

  ## EXAMPLES

  To suspend Policy Controller, run:

    $ {command}

  To re-enable Policy Controller webhooks, use the `enable` command:

    $ {parent_command} enable
  """
    feature_name = 'policycontroller'

    @classmethod
    def Args(cls, parser):
        cmd_flags = flags.PocoFlags(parser, 'suspend')
        cmd_flags.add_memberships()

    def Run(self, args):
        specs = self.path_specs(args)
        updated_specs = {path: self.suspend(spec) for path, spec in specs.items()}
        return self.update_specs(updated_specs)

    def suspend(self, spec):
        """Sets the membership spec to SUSPENDED.

    Args:
      spec: The spec to be suspended.

    Returns:
      Updated spec, based on message api version.
    """
        spec.policycontroller.policyControllerHubConfig.installSpec = self.messages.PolicyControllerHubConfig.InstallSpecValueValuesEnum.INSTALL_SPEC_SUSPENDED
        return spec