from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class StartAsyncReplication(base.Command):
    """Start Async Replication on Compute Engine persistent disks."""

    @classmethod
    def Args(cls, parser):
        StartAsyncReplication.disks_arg = disks_flags.MakeDiskArg(plural=False)
        StartAsyncReplication.secondary_disk_arg = disks_flags.MakeSecondaryDiskArg(required=True)
        _CommonArgs(parser)

    def GetAsyncSecondaryDiskUri(self, args, compute_holder):
        secondary_disk_ref = None
        if args.secondary_disk:
            secondary_disk_project = getattr(args, 'secondary_disk_project', None)
            secondary_disk_ref = self.secondary_disk_arg.ResolveAsResource(args, compute_holder.resources, source_project=secondary_disk_project)
            if secondary_disk_ref:
                return secondary_disk_ref.SelfLink()
        return None

    @classmethod
    def _GetApiHolder(cls, no_http=False):
        return base_classes.ComputeApiHolder(cls.ReleaseTrack(), no_http)

    def Run(self, args):
        return self._Run(args)

    def _Run(self, args):
        compute_holder = self._GetApiHolder()
        client = compute_holder.client
        disk_ref = StartAsyncReplication.disks_arg.ResolveAsResource(args, compute_holder.resources, scope_lister=flags.GetDefaultScopeLister(client))
        request = None
        secondary_disk_uri = self.GetAsyncSecondaryDiskUri(args, compute_holder)
        if disk_ref.Collection() == 'compute.disks':
            request = client.messages.ComputeDisksStartAsyncReplicationRequest(disk=disk_ref.Name(), project=disk_ref.project, zone=disk_ref.zone, disksStartAsyncReplicationRequest=client.messages.DisksStartAsyncReplicationRequest(asyncSecondaryDisk=secondary_disk_uri))
            request = (client.apitools_client.disks, 'StartAsyncReplication', request)
        elif disk_ref.Collection() == 'compute.regionDisks':
            request = client.messages.ComputeRegionDisksStartAsyncReplicationRequest(disk=disk_ref.Name(), project=disk_ref.project, region=disk_ref.region, regionDisksStartAsyncReplicationRequest=client.messages.RegionDisksStartAsyncReplicationRequest(asyncSecondaryDisk=secondary_disk_uri))
            request = (client.apitools_client.regionDisks, 'StartAsyncReplication', request)
        return client.MakeRequests([request])