from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import conversion_workspaces
from googlecloudsdk.api_lib.database_migration import resource_args
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.database_migration.conversion_workspaces import flags as cw_flags
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Apply(base.Command):
    """Apply a Database Migration Service conversion workspace."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        resource_args.AddConversionWorkspaceApplyResourceArg(parser, 'to apply')
        cw_flags.AddNoAsyncFlag(parser)
        cw_flags.AddFilterFlag(parser)

    def Run(self, args):
        """Apply a Database Migration Service conversion workspace.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      A dict object representing the operations resource describing the apply
      operation if the apply was successful.
    """
        conversion_workspace_ref = args.CONCEPTS.conversion_workspace.Parse()
        destination_connection_profile_ref = args.CONCEPTS.destination_connection_profile.Parse()
        cw_client = conversion_workspaces.ConversionWorkspacesClient(self.ReleaseTrack())
        result_operation = cw_client.Apply(conversion_workspace_ref.RelativeName(), destination_connection_profile_ref, args)
        client = api_util.GetClientInstance(self.ReleaseTrack())
        messages = api_util.GetMessagesModule(self.ReleaseTrack())
        resource_parser = api_util.GetResourceParser(self.ReleaseTrack())
        if args.IsKnownAndSpecified('no_async'):
            log.status.Print('Waiting for conversion workspace [{}] to be applied with [{}]'.format(conversion_workspace_ref.conversionWorkspacesId, result_operation.name))
            api_util.HandleLRO(client, result_operation, client.projects_locations_conversionWorkspaces)
            log.status.Print('Applied conversion workspace {} [{}]'.format(conversion_workspace_ref.conversionWorkspacesId, result_operation.name))
            return
        operation_ref = resource_parser.Create('datamigration.projects.locations.operations', operationsId=result_operation.name, projectsId=conversion_workspace_ref.projectsId, locationsId=conversion_workspace_ref.locationsId)
        return client.projects_locations_operations.Get(messages.DatamigrationProjectsLocationsOperationsGetRequest(name=operation_ref.operationsId))