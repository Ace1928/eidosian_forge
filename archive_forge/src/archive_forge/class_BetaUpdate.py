from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import config_changes
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import container_parser
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import messages_util
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class BetaUpdate(Update):
    """Update a Cloud Run Job."""

    @classmethod
    def Args(cls, parser):
        Update.CommonArgs(parser)
        flags.AddVpcNetworkGroupFlagsForUpdate(parser, resource_kind='job')
        flags.AddVolumesFlags(parser, cls.ReleaseTrack())
        group = base.ArgumentGroup()
        group.AddArgument(flags.AddVolumeMountFlag())
        group.AddArgument(flags.RemoveVolumeMountFlag())
        group.AddArgument(flags.ClearVolumeMountsFlag())
        group.AddToParser(parser)