from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.command_lib.container import container_command_util as cmd_util
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
def ParseCreateNodePoolOptions(self, args):
    ops = ParseCreateNodePoolOptionsBase(args)
    flags.WarnForNodeVersionAutoUpgrade(args)
    flags.ValidateSurgeUpgradeSettings(args)
    ephemeral_storage = None
    if args.IsKnownAndSpecified('ephemeral_storage'):
        ephemeral_storage = [] if args.ephemeral_storage is None else args.ephemeral_storage
    ops.local_ssd_volume_configs = args.local_ssd_volumes
    ops.boot_disk_kms_key = args.boot_disk_kms_key
    ops.sandbox = args.sandbox
    ops.linux_sysctls = args.linux_sysctls
    ops.node_locations = args.node_locations
    ops.system_config_from_file = args.system_config_from_file
    ops.enable_gcfs = args.enable_gcfs
    ops.enable_image_streaming = args.enable_image_streaming
    ops.spot = args.spot
    ops.placement_type = args.placement_type
    ops.placement_policy = args.placement_policy
    ops.location_policy = args.location_policy
    ops.enable_blue_green_upgrade = args.enable_blue_green_upgrade
    ops.enable_surge_upgrade = args.enable_surge_upgrade
    ops.node_pool_soak_duration = args.node_pool_soak_duration
    ops.standard_rollout_policy = args.standard_rollout_policy
    ops.maintenance_interval = args.maintenance_interval
    ops.network_performance_config = args.network_performance_configs
    ops.enable_confidential_nodes = args.enable_confidential_nodes
    ops.disable_pod_cidr_overprovision = args.disable_pod_cidr_overprovision
    ops.enable_fast_socket = args.enable_fast_socket
    ops.enable_queued_provisioning = args.enable_queued_provisioning
    ops.tpu_topology = args.tpu_topology
    ops.enable_nested_virtualization = args.enable_nested_virtualization
    ops.enable_best_effort_provision = args.enable_best_effort_provision
    ops.min_provision_nodes = args.min_provision_nodes
    ops.host_maintenance_interval = args.host_maintenance_interval
    ops.performance_monitoring_unit = args.performance_monitoring_unit
    ops.autoscaled_rollout_policy = args.autoscaled_rollout_policy
    ops.ephemeral_storage = ephemeral_storage
    ops.secondary_boot_disks = args.secondary_boot_disk
    ops.enable_confidential_storage = args.enable_confidential_storage
    ops.storage_pools = args.storage_pools
    return ops