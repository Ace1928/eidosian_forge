from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import connection_profiles
from googlecloudsdk.api_lib.database_migration import resource_args
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
class _List(object):
    """Base class for listing Database Migration Service connection profiles."""

    @classmethod
    def Args(cls, parser):
        """Register flags for this command."""
        concept_parsers.ConceptParser.ForResource('--region', resource_args.GetRegionResourceSpec(), group_help='The location you want to list the connection profiles for.').AddToParser(parser)
        parser.display_info.AddFormat("\n          table(\n            name.basename():label=CONNECTION_PROFILE_ID,\n            display_name,\n            name.scope('locations').segment(0):label=REGION,\n            state,\n            provider_display:label=PROVIDER,\n            engine,\n            host:label=HOSTNAME/IP,\n            create_time.date():label=CREATED\n          )\n        ")
        parser.display_info.AddUriFunc(lambda p: _GetUri(cls.ReleaseTrack(), p))

    def Run(self, args):
        """Runs the command.

    Args:
      args: All the arguments that were provided to this command invocation.

    Returns:
      An iterator over objects containing connection profile data.
    """
        cp_client = connection_profiles.ConnectionProfilesClient(self.ReleaseTrack())
        project_id = properties.VALUES.core.project.Get(required=True)
        profiles = cp_client.List(project_id, args)
        if args.format is None or args.format.startswith('"table'):
            return [_ConnectionProfileInfo(profile, self._GetHost(profile), cp_client.GetEngineName(profile)) for profile in profiles]
        return profiles