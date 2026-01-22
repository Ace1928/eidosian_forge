from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import name_generator
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
from googlecloudsdk.command_lib.compute.snapshots import flags as snap_flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves import zip
@base.ReleaseTracks(base.ReleaseTrack.GA)
class SnapshotDisks(base.SilentCommand):
    """Create snapshots of Google Compute Engine persistent disks."""

    @classmethod
    def Args(cls, parser):
        SnapshotDisks.disks_arg = disks_flags.MakeDiskArg(plural=True)
        labels_util.AddCreateLabelsFlags(parser)
        _CommonArgs(parser)

    def Run(self, args):
        return self._Run(args)

    def _Run(self, args):
        """Returns a list of requests necessary for snapshotting disks."""
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        disk_refs = SnapshotDisks.disks_arg.ResolveAsResource(args, holder.resources, scope_lister=flags.GetDefaultScopeLister(holder.client))
        if args.snapshot_names:
            if len(disk_refs) != len(args.snapshot_names):
                raise exceptions.InvalidArgumentException('--snapshot-names', '[--snapshot-names] must have the same number of values as disks being snapshotted.')
            snapshot_names = args.snapshot_names
        else:
            snapshot_names = [name_generator.GenerateRandomName() for _ in disk_refs]
        snapshot_refs = [holder.resources.Parse(snapshot_name, params={'project': properties.VALUES.core.project.GetOrFail}, collection='compute.snapshots') for snapshot_name in snapshot_names]
        client = holder.client.apitools_client
        messages = holder.client.messages
        requests = []
        for disk_ref, snapshot_ref in zip(disk_refs, snapshot_refs):
            csek_keys = csek_utils.CsekKeyStore.FromArgs(args, True)
            snapshot_key_or_none = csek_utils.MaybeLookupKeyMessage(csek_keys, snapshot_ref, client)
            disk_key_or_none = csek_utils.MaybeLookupKeyMessage(csek_keys, disk_ref, client)
            snapshot_message = messages.Snapshot(name=snapshot_ref.Name(), description=args.description, snapshotEncryptionKey=snapshot_key_or_none, sourceDiskEncryptionKey=disk_key_or_none, chainName=args.chain_name)
            if hasattr(args, 'storage_location') and args.IsSpecified('storage_location'):
                snapshot_message.storageLocations = [args.storage_location]
            if hasattr(args, 'labels') and args.IsSpecified('labels'):
                snapshot_message.labels = labels_util.ParseCreateArgs(args, messages.Snapshot.LabelsValue)
            if disk_ref.Collection() == 'compute.disks':
                request = messages.ComputeDisksCreateSnapshotRequest(disk=disk_ref.Name(), snapshot=snapshot_message, project=disk_ref.project, zone=disk_ref.zone, guestFlush=args.guest_flush)
                requests.append((client.disks, 'CreateSnapshot', request))
            elif disk_ref.Collection() == 'compute.regionDisks':
                request = messages.ComputeRegionDisksCreateSnapshotRequest(disk=disk_ref.Name(), snapshot=snapshot_message, project=disk_ref.project, region=disk_ref.region)
                if hasattr(request, 'guestFlush'):
                    guest_flush = getattr(args, 'guest_flush', None)
                    if guest_flush is not None:
                        request.guestFlush = guest_flush
                requests.append((client.regionDisks, 'CreateSnapshot', request))
        errors_to_collect = []
        responses = holder.client.AsyncRequests(requests, errors_to_collect)
        for r in responses:
            err = getattr(r, 'error', None)
            if err:
                errors_to_collect.append(poller.OperationErrors(err.errors))
        if errors_to_collect:
            raise core_exceptions.MultiError(errors_to_collect)
        operation_refs = [holder.resources.Parse(r.selfLink) for r in responses]
        if args.async_:
            for operation_ref in operation_refs:
                log.status.Print('Disk snapshot in progress for [{}].'.format(operation_ref.SelfLink()))
            log.status.Print('Use [gcloud compute operations describe URI] command to check the status of the operation(s).')
            return responses
        operation_poller = poller.BatchPoller(holder.client, client.snapshots, snapshot_refs)
        return waiter.WaitFor(operation_poller, poller.OperationBatch(operation_refs), 'Creating snapshot(s) {0}'.format(', '.join((s.Name() for s in snapshot_refs))), max_wait_ms=None)