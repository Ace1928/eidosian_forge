from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils as compute_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags
from googlecloudsdk.command_lib.compute.resource_policies import util
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class UpdateSnapshotScheduleBeta(UpdateSnapshotSchedule):
    """Update a Compute Engine Snapshot Schedule Resource Policy."""

    @staticmethod
    def Args(parser):
        _CommonArgs(parser, compute_api.COMPUTE_BETA_API_VERSION)

    def Run(self, args):
        return self._Run(args)