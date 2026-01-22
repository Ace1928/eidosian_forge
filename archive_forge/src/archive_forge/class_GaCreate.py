from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.base import ReleaseTrack
from googlecloudsdk.command_lib.assured import create_workload
from googlecloudsdk.command_lib.assured import flags
@base.ReleaseTracks(ReleaseTrack.GA)
class GaCreate(create_workload.CreateWorkload):
    """Create a new Assured Workloads environment."""

    @staticmethod
    def Args(parser):
        flags.AddCreateWorkloadFlags(parser, ReleaseTrack.GA)