from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.GA)
class UpdateContainer(base.UpdateCommand):
    """Command for updating VM instances running container images."""

    @staticmethod
    def Args(parser):
        """Register parser args."""
        instances_flags.AddUpdateContainerArgs(parser, container_mount_disk_enabled=True)

    def Run(self, args):
        """Issues requests necessary to update Container."""
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        instance_ref = instances_flags.INSTANCE_ARG.ResolveAsResource(args, holder.resources, scope_lister=instances_flags.GetInstanceZoneScopeLister(client))
        instance = client.apitools_client.instances.Get(client.messages.ComputeInstancesGetRequest(**instance_ref.AsDict()))
        container_mount_disk = instances_flags.GetValidatedContainerMountDisk(holder, args.container_mount_disk, instance.disks, [], for_update=True, client=client.apitools_client)
        containers_utils.UpdateInstance(holder, client, instance_ref, instance, args, container_mount_disk_enabled=True, container_mount_disk=container_mount_disk)