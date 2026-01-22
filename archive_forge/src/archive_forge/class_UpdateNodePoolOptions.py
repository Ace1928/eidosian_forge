from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
class UpdateNodePoolOptions(object):
    """Options to pass to UpdateNodePool."""

    def __init__(self, enable_autorepair=None, enable_autoupgrade=None, enable_autoscaling=None, max_nodes=None, min_nodes=None, total_max_nodes=None, total_min_nodes=None, location_policy=None, enable_autoprovisioning=None, workload_metadata=None, workload_metadata_from_node=None, node_locations=None, max_surge_upgrade=None, max_unavailable_upgrade=None, system_config_from_file=None, node_labels=None, labels=None, node_taints=None, tags=None, enable_private_nodes=None, enable_gcfs=None, gvnic=None, enable_image_streaming=None, enable_blue_green_upgrade=None, enable_surge_upgrade=None, node_pool_soak_duration=None, standard_rollout_policy=None, autoscaled_rollout_policy=None, network_performance_config=None, enable_confidential_nodes=None, enable_fast_socket=None, logging_variant=None, accelerators=None, windows_os_version=None, enable_insecure_kubelet_readonly_port=None, resource_manager_tags=None, containerd_config_from_file=None, secondary_boot_disks=None, machine_type=None, disk_type=None, disk_size_gb=None, enable_queued_provisioning=None):
        self.enable_autorepair = enable_autorepair
        self.enable_autoupgrade = enable_autoupgrade
        self.enable_autoscaling = enable_autoscaling
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.accelerators = accelerators
        self.total_max_nodes = total_max_nodes
        self.total_min_nodes = total_min_nodes
        self.location_policy = location_policy
        self.enable_autoprovisioning = enable_autoprovisioning
        self.workload_metadata = workload_metadata
        self.workload_metadata_from_node = workload_metadata_from_node
        self.node_locations = node_locations
        self.max_surge_upgrade = max_surge_upgrade
        self.max_unavailable_upgrade = max_unavailable_upgrade
        self.system_config_from_file = system_config_from_file
        self.labels = labels
        self.node_labels = node_labels
        self.node_taints = node_taints
        self.tags = tags
        self.enable_private_nodes = enable_private_nodes
        self.enable_gcfs = enable_gcfs
        self.gvnic = gvnic
        self.enable_image_streaming = enable_image_streaming
        self.enable_blue_green_upgrade = enable_blue_green_upgrade
        self.enable_surge_upgrade = enable_surge_upgrade
        self.node_pool_soak_duration = node_pool_soak_duration
        self.standard_rollout_policy = standard_rollout_policy
        self.autoscaled_rollout_policy = autoscaled_rollout_policy
        self.network_performance_config = network_performance_config
        self.enable_confidential_nodes = enable_confidential_nodes
        self.enable_fast_socket = enable_fast_socket
        self.logging_variant = logging_variant
        self.windows_os_version = windows_os_version
        self.enable_insecure_kubelet_readonly_port = enable_insecure_kubelet_readonly_port
        self.resource_manager_tags = resource_manager_tags
        self.containerd_config_from_file = containerd_config_from_file
        self.secondary_boot_disks = secondary_boot_disks
        self.machine_type = machine_type
        self.disk_type = disk_type
        self.disk_size_gb = disk_size_gb
        self.enable_queued_provisioning = enable_queued_provisioning
        self.enable_insecure_kubelet_readonly_port = enable_insecure_kubelet_readonly_port

    def IsAutoscalingUpdate(self):
        return self.enable_autoscaling is not None or self.max_nodes is not None or self.min_nodes is not None or (self.total_max_nodes is not None) or (self.total_min_nodes is not None) or (self.enable_autoprovisioning is not None) or (self.location_policy is not None)

    def IsNodePoolManagementUpdate(self):
        return self.enable_autorepair is not None or self.enable_autoupgrade is not None

    def IsUpdateNodePoolRequest(self):
        return self.workload_metadata is not None or self.workload_metadata_from_node is not None or self.node_locations is not None or (self.max_surge_upgrade is not None) or (self.max_unavailable_upgrade is not None) or (self.system_config_from_file is not None) or (self.labels is not None) or (self.node_labels is not None) or (self.node_taints is not None) or (self.tags is not None) or (self.enable_private_nodes is not None) or (self.enable_gcfs is not None) or (self.gvnic is not None) or (self.enable_image_streaming is not None) or (self.enable_surge_upgrade is not None) or (self.enable_blue_green_upgrade is not None) or (self.node_pool_soak_duration is not None) or (self.standard_rollout_policy is not None) or (self.network_performance_config is not None) or (self.enable_confidential_nodes is not None) or (self.enable_fast_socket is not None) or (self.logging_variant is not None) or (self.windows_os_version is not None) or (self.accelerators is not None) or (self.resource_manager_tags is not None) or (self.containerd_config_from_file is not None) or (self.machine_type is not None) or (self.disk_type is not None) or (self.disk_size_gb is not None) or (self.enable_queued_provisioning is not None)