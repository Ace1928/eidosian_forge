from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.workbench import instances as instance_util
from googlecloudsdk.api_lib.workbench import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.workbench import flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class IsUpgradeable(base.DescribeCommand):
    """Checks if a workbench instance is upgradeable."""

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        flags.AddIsUpgradeableInstanceFlags(parser)

    def Run(self, args):
        release_track = self.ReleaseTrack()
        client = util.GetClient(release_track)
        messages = util.GetMessages(release_track)
        instance_service = client.projects_locations_instances
        result = instance_service.CheckUpgradability(instance_util.CreateInstanceCheckUpgradabilityRequest(args, messages))
        return result