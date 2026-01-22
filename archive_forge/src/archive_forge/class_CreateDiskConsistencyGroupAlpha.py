from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags
from googlecloudsdk.command_lib.compute.resource_policies import util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class CreateDiskConsistencyGroupAlpha(CreateDiskConsistencyGroup):
    """Create a Compute Engine Disk Consistency Group resource policy."""

    @staticmethod
    def Args(parser):
        CreateDiskConsistencyGroup.resource_policy_arg = flags.MakeResourcePolicyArg()
        _CommonArgs(parser)

    def Run(self, args):
        return self._Run(args)