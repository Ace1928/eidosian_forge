from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def ParseUpdateNodePoolOptions(self, args):
    flags.ValidateSurgeUpgradeSettings(args)
    ops = api_adapter.UpdateNodePoolOptions(accelerators=args.accelerator, enable_autorepair=args.enable_autorepair, enable_autoupgrade=args.enable_autoupgrade, enable_autoscaling=args.enable_autoscaling, max_nodes=args.max_nodes, min_nodes=args.min_nodes, total_max_nodes=args.total_max_nodes, total_min_nodes=args.total_min_nodes, location_policy=args.location_policy, enable_autoprovisioning=args.enable_autoprovisioning, workload_metadata=args.workload_metadata, workload_metadata_from_node=args.workload_metadata_from_node, node_locations=args.node_locations, max_surge_upgrade=args.max_surge_upgrade, max_unavailable_upgrade=args.max_unavailable_upgrade, system_config_from_file=args.system_config_from_file, labels=args.labels, node_labels=args.node_labels, node_taints=args.node_taints, tags=args.tags, enable_private_nodes=args.enable_private_nodes, enable_gcfs=args.enable_gcfs, gvnic=args.enable_gvnic, enable_image_streaming=args.enable_image_streaming, enable_blue_green_upgrade=args.enable_blue_green_upgrade, enable_surge_upgrade=args.enable_surge_upgrade, node_pool_soak_duration=args.node_pool_soak_duration, standard_rollout_policy=args.standard_rollout_policy, autoscaled_rollout_policy=args.autoscaled_rollout_policy, network_performance_config=args.network_performance_configs, enable_confidential_nodes=args.enable_confidential_nodes, enable_fast_socket=args.enable_fast_socket, logging_variant=args.logging_variant, windows_os_version=args.windows_os_version, resource_manager_tags=args.resource_manager_tags, containerd_config_from_file=args.containerd_config_from_file, enable_queued_provisioning=args.enable_queued_provisioning, machine_type=args.machine_type, disk_type=args.disk_type, enable_insecure_kubelet_readonly_port=args.enable_insecure_kubelet_readonly_port, disk_size_gb=utils.BytesToGb(args.disk_size) if hasattr(args, 'disk_size') else None)
    return ops