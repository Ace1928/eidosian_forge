from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.workbench import instances as instance_util
from googlecloudsdk.api_lib.workbench import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.workbench import flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Diagnose(base.Command):
    """Diagnoses a workbench instance."""

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        flags.AddDiagnoseInstanceFlags(parser)

    def Run(self, args):
        """This is what gets called when the user runs this command."""
        release_track = self.ReleaseTrack()
        client = util.GetClient(release_track)
        messages = util.GetMessages(release_track)
        instance_service = client.projects_locations_instances
        operation = instance_service.Diagnose(instance_util.CreateInstanceDiagnoseRequest(args, messages))
        return instance_util.HandleLRO(operation, args, instance_service, release_track, operation_type=instance_util.OperationType.UPDATE)