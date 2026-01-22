from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute.instances.create import utils as create_utils
from googlecloudsdk.command_lib.compute import resource_manager_tags_utils
from googlecloudsdk.command_lib.compute import secure_tags_utils
from googlecloudsdk.command_lib.compute.resource_policies import util as maintenance_util
from googlecloudsdk.command_lib.util.apis import arg_utils
import six
def CreateBulkInsertInstanceResource(args, holder, compute_client, resource_parser, project, location, scope, instance_template_resource, supported_features):
    """Create bulkInsertInstanceResource."""
    name_pattern = args.name_pattern
    instance_names = args.predefined_names or []
    instance_count = args.count or len(instance_names)
    per_instance_props = _GetPerInstanceProperties(args, compute_client.messages, instance_names, supported_features.support_custom_hostnames)
    location_policy = _GetLocationPolicy(args, compute_client.messages, supported_features.support_enable_target_shape, supported_features.support_max_count_per_zone)
    instance_min_count = instance_count
    if args.IsSpecified('min_count'):
        instance_min_count = args.min_count
    source_instance_template = _GetSourceInstanceTemplate(args, resource_parser, instance_template_resource)
    skip_defaults = source_instance_template is not None
    scheduling = instance_utils.GetScheduling(args, compute_client, skip_defaults, support_node_affinity=False, support_min_node_cpu=supported_features.support_min_node_cpu, support_host_error_timeout_seconds=supported_features.support_host_error_timeout_seconds, support_max_run_duration=supported_features.support_max_run_duration, support_local_ssd_recovery_timeout=supported_features.support_local_ssd_recovery_timeout)
    tags = instance_utils.GetTags(args, compute_client)
    labels = instance_utils.GetLabels(args, compute_client, instance_properties=True)
    metadata = instance_utils.GetMetadata(args, compute_client, skip_defaults)
    network_interfaces = create_utils.GetBulkNetworkInterfaces(args=args, resource_parser=resource_parser, compute_client=compute_client, holder=holder, project=project, location=location, scope=scope, skip_defaults=skip_defaults)
    create_boot_disk = not instance_utils.UseExistingBootDisk((args.disk or []) + (args.create_disk or []))
    image_uri = create_utils.GetImageUri(args, compute_client, create_boot_disk, project, resource_parser)
    shielded_instance_config = create_utils.BuildShieldedInstanceConfigMessage(messages=compute_client.messages, args=args)
    confidential_vm_type = None
    if supported_features.support_confidential_compute:
        confidential_instance_config = create_utils.BuildConfidentialInstanceConfigMessage(messages=compute_client.messages, args=args, support_confidential_compute_type=supported_features.support_confidential_compute_type, support_confidential_compute_type_tdx=supported_features.support_confidential_compute_type_tdx)
        confidential_vm_type = instance_utils.GetConfidentialVmType(args, supported_features.support_confidential_compute_type)
    service_accounts = create_utils.GetProjectServiceAccount(args, project, compute_client, skip_defaults)
    boot_disk_size_gb = instance_utils.GetBootDiskSizeGb(args)
    disks = []
    if create_utils.CheckSpecifiedDiskArgs(args=args, support_disks=False, skip_defaults=skip_defaults):
        for disk in args.disk or []:
            disk['mode'] = 'ro'
        disks = create_utils.CreateDiskMessages(args=args, project=project, location=location, scope=scope, compute_client=compute_client, resource_parser=resource_parser, image_uri=image_uri, create_boot_disk=create_boot_disk, boot_disk_size_gb=boot_disk_size_gb, support_kms=True, support_nvdimm=supported_features.support_nvdimm, support_source_snapshot_csek=supported_features.support_source_snapshot_csek, support_boot_snapshot_uri=supported_features.support_boot_snapshot_uri, support_image_csek=supported_features.support_image_csek, support_create_disk_snapshots=supported_features.support_create_disk_snapshots, use_disk_type_uri=False)
    machine_type_name = None
    if instance_utils.CheckSpecifiedMachineTypeArgs(args, skip_defaults):
        machine_type_name = instance_utils.CreateMachineTypeName(args, confidential_vm_type)
    can_ip_forward = instance_utils.GetCanIpForward(args, skip_defaults)
    guest_accelerators = create_utils.GetAcceleratorsForInstanceProperties(args=args, compute_client=compute_client)
    advanced_machine_features = None
    if args.enable_nested_virtualization is not None or args.threads_per_core is not None or (supported_features.support_numa_node_count and args.numa_node_count is not None) or (supported_features.support_visible_core_count and args.visible_core_count is not None) or (args.enable_uefi_networking is not None) or (supported_features.support_performance_monitoring_unit and args.performance_monitoring_unit) or (supported_features.support_watchdog_timer and args.enable_watchdog_timer is not None):
        visible_core_count = args.visible_core_count if supported_features.support_visible_core_count else None
        advanced_machine_features = instance_utils.CreateAdvancedMachineFeaturesMessage(compute_client.messages, args.enable_nested_virtualization, args.threads_per_core, args.numa_node_count if supported_features.support_numa_node_count else None, visible_core_count, args.enable_uefi_networking, args.performance_monitoring_unit if supported_features.support_performance_monitoring_unit else None, args.enable_watchdog_timer if supported_features.support_watchdog_timer else None)
    parsed_resource_policies = []
    resource_policies = getattr(args, 'resource_policies', None)
    if resource_policies:
        for policy in resource_policies:
            resource_policy_ref = maintenance_util.ParseResourcePolicyWithScope(resource_parser, policy, project=project, location=location, scope=scope)
            parsed_resource_policies.append(resource_policy_ref.Name())
    display_device = None
    if supported_features.support_display_device and args.IsSpecified('enable_display_device'):
        display_device = compute_client.messages.DisplayDevice(enableDisplay=args.enable_display_device)
    reservation_affinity = instance_utils.GetReservationAffinity(args, compute_client, supported_features.support_specific_then_x_affinity)
    instance_properties = compute_client.messages.InstanceProperties(canIpForward=can_ip_forward, description=args.description, disks=disks, guestAccelerators=guest_accelerators, labels=labels, machineType=machine_type_name, metadata=metadata, minCpuPlatform=args.min_cpu_platform, networkInterfaces=network_interfaces, serviceAccounts=service_accounts, scheduling=scheduling, tags=tags, resourcePolicies=parsed_resource_policies, shieldedInstanceConfig=shielded_instance_config, reservationAffinity=reservation_affinity, advancedMachineFeatures=advanced_machine_features)
    if supported_features.support_secure_tags and args.secure_tags:
        instance_properties.secureTags = secure_tags_utils.GetSecureTags(args.secure_tags)
    if args.resource_manager_tags:
        ret_resource_manager_tags = resource_manager_tags_utils.GetResourceManagerTags(args.resource_manager_tags)
        if ret_resource_manager_tags is not None:
            properties_message = compute_client.messages.InstanceProperties
            instance_properties.resourceManagerTags = properties_message.ResourceManagerTagsValue(additionalProperties=[properties_message.ResourceManagerTagsValue.AdditionalProperty(key=key, value=value) for key, value in sorted(six.iteritems(ret_resource_manager_tags))])
    if supported_features.support_display_device and display_device:
        instance_properties.displayDevice = display_device
    if supported_features.support_confidential_compute and confidential_instance_config:
        instance_properties.confidentialInstanceConfig = confidential_instance_config
    if supported_features.support_erase_vss and args.IsSpecified('erase_windows_vss_signature'):
        instance_properties.eraseWindowsVssSignature = args.erase_windows_vss_signature
    if supported_features.support_post_key_revocation_action_type and args.IsSpecified('post_key_revocation_action_type'):
        instance_properties.postKeyRevocationActionType = arg_utils.ChoiceToEnum(args.post_key_revocation_action_type, compute_client.messages.Instance.PostKeyRevocationActionTypeValueValuesEnum)
    if args.IsSpecified('network_performance_configs'):
        instance_properties.networkPerformanceConfig = instance_utils.GetNetworkPerformanceConfig(args, compute_client)
    return compute_client.messages.BulkInsertInstanceResource(count=instance_count, instanceProperties=instance_properties, minCount=instance_min_count, perInstanceProperties=per_instance_props, sourceInstanceTemplate=source_instance_template, namePattern=name_pattern, locationPolicy=location_policy)