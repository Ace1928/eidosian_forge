from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class StopAsyncReplication(base.Command):
    """Stop Async Replication on Compute Engine persistent disks."""

    @classmethod
    def Args(cls, parser):
        StopAsyncReplication.disks_arg = disks_flags.MakeDiskArg(plural=False)
        _CommonArgs(parser)

    @classmethod
    def _GetApiHolder(cls, no_http=False):
        return base_classes.ComputeApiHolder(cls.ReleaseTrack(), no_http)

    def Run(self, args):
        return self._Run(args)

    def _Run(self, args):
        compute_holder = self._GetApiHolder()
        client = compute_holder.client
        disk_ref = StopAsyncReplication.disks_arg.ResolveAsResource(args, compute_holder.resources, scope_lister=flags.GetDefaultScopeLister(client))
        request = None
        if disk_ref.Collection() == 'compute.disks':
            request = client.messages.ComputeDisksStopAsyncReplicationRequest(disk=disk_ref.Name(), project=disk_ref.project, zone=disk_ref.zone)
            request = (client.apitools_client.disks, 'StopAsyncReplication', request)
        elif disk_ref.Collection() == 'compute.regionDisks':
            request = client.messages.ComputeRegionDisksStopAsyncReplicationRequest(disk=disk_ref.Name(), project=disk_ref.project, region=disk_ref.region)
            request = (client.apitools_client.regionDisks, 'StopAsyncReplication', request)
        return client.MakeRequests([request])