from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.GA)
class StopGroupAsyncReplication(base.Command):
    """Stop Group Async Replication for a Consistency Group Resource Policy."""

    @classmethod
    def Args(cls, parser):
        _CommonArgs(parser)

    @classmethod
    def _GetApiHolder(cls, no_http=False):
        return base_classes.ComputeApiHolder(cls.ReleaseTrack(), no_http)

    def Run(self, args):
        return self._Run(args)

    def _Run(self, args):
        compute_holder = self._GetApiHolder()
        client = compute_holder.client
        policy_url = getattr(args, 'DISK_CONSISTENCY_GROUP_POLICY', None)
        project = properties.VALUES.core.project.GetOrFail()
        if args.IsSpecified('zone'):
            request = client.messages.ComputeDisksStopGroupAsyncReplicationRequest(project=project, zone=args.zone, disksStopGroupAsyncReplicationResource=client.messages.DisksStopGroupAsyncReplicationResource(resourcePolicy=policy_url))
            request = (client.apitools_client.disks, 'StopGroupAsyncReplication', request)
        else:
            request = client.messages.ComputeRegionDisksStopGroupAsyncReplicationRequest(project=project, region=args.region, disksStopGroupAsyncReplicationResource=client.messages.DisksStopGroupAsyncReplicationResource(resourcePolicy=policy_url))
            request = (client.apitools_client.regionDisks, 'StopGroupAsyncReplication', request)
        return client.MakeRequests([request], no_followup=True)