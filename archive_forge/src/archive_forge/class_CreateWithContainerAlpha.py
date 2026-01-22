from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_template_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.instances.create import utils as create_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_templates import flags as instance_templates_flags
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class CreateWithContainerAlpha(CreateWithContainerBeta):
    """Command for creating VM instance templates hosting Docker images."""

    @staticmethod
    def Args(parser):
        _Args(parser, base.ReleaseTrack.ALPHA, container_mount_enabled=True, enable_guest_accelerators=True, support_region_instance_template=True)
        instances_flags.AddLocalNvdimmArgs(parser)
        instances_flags.AddPrivateIpv6GoogleAccessArgForTemplate(parser, utils.COMPUTE_ALPHA_API_VERSION)
        instances_flags.AddStackTypeArgs(parser)
        instances_flags.AddIpv6NetworkTierArgs(parser)
        instances_flags.AddIPv6AddressAlphaArgs(parser)
        instances_flags.AddIPv6PrefixLengthAlphaArgs(parser)
        instances_flags.AddInternalIPv6AddressArgs(parser)
        instances_flags.AddInternalIPv6PrefixLengthArgs(parser)

    def Run(self, args):
        """Issues an InstanceTemplates.Insert request.

    Args:
      args: the argparse arguments that this command was invoked with.

    Returns:
      an InstanceTemplate message object
    """
        self._ValidateArgs(args)
        instances_flags.ValidateNetworkTierArgs(args)
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        container_mount_disk = instances_flags.GetValidatedContainerMountDisk(holder, args.container_mount_disk, args.disk, args.create_disk)
        client = holder.client
        instance_template_ref = self._GetInstanceTemplateRef(args, holder)
        image_uri = self._GetImageUri(args, client, holder, instance_template_ref)
        labels = containers_utils.GetLabelsMessageWithCosVersion(None, image_uri, holder.resources, client.messages.InstanceProperties)
        argument_labels = labels_util.ParseCreateArgs(args, client.messages.InstanceProperties.LabelsValue)
        if argument_labels:
            labels.additionalProperties.extend(argument_labels.additionalProperties)
        metadata = self._GetUserMetadata(args, client, instance_template_ref, container_mount_disk_enabled=True, container_mount_disk=container_mount_disk)
        network_interfaces = self._GetNetworkInterfaces(args, client, holder)
        scheduling = self._GetScheduling(args, client)
        service_accounts = self._GetServiceAccounts(args, client)
        machine_type = self._GetMachineType(args)
        disks = self._GetDisks(args, client, holder, instance_template_ref, image_uri, match_container_mount_disks=True)
        confidential_instance_config = create_utils.BuildConfidentialInstanceConfigMessage(messages=client.messages, args=args, support_confidential_compute_type=True, support_confidential_compute_type_tdx=True)
        guest_accelerators = instance_template_utils.CreateAcceleratorConfigMessages(client.messages, getattr(args, 'accelerator', None))
        properties = client.messages.InstanceProperties(machineType=machine_type, disks=disks, canIpForward=args.can_ip_forward, confidentialInstanceConfig=confidential_instance_config, labels=labels, metadata=metadata, minCpuPlatform=args.min_cpu_platform, networkInterfaces=network_interfaces, serviceAccounts=service_accounts, scheduling=scheduling, tags=containers_utils.CreateTagsMessage(client.messages, args.tags), guestAccelerators=guest_accelerators)
        if args.private_ipv6_google_access_type is not None:
            properties.privateIpv6GoogleAccess = instances_flags.GetPrivateIpv6GoogleAccessTypeFlagMapperForTemplate(client.messages).GetEnumForChoice(args.private_ipv6_google_access_type)
        shielded_instance_config_message = create_utils.BuildShieldedInstanceConfigMessage(messages=client.messages, args=args)
        if shielded_instance_config_message:
            properties.shieldedInstanceConfig = shielded_instance_config_message
        request = client.messages.ComputeInstanceTemplatesInsertRequest(instanceTemplate=client.messages.InstanceTemplate(properties=properties, description=args.description, name=instance_template_ref.Name()), project=instance_template_ref.project)
        return client.MakeRequests([(client.apitools_client.instanceTemplates, 'Insert', request)])