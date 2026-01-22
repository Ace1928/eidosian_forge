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
def CreateNodePoolCommon(self, node_pool_ref, options):
    """Returns a CreateNodePool operation."""
    node_config = self.messages.NodeConfig()
    if options.machine_type:
        node_config.machineType = options.machine_type
    if options.disk_size_gb:
        node_config.diskSizeGb = options.disk_size_gb
    if options.disk_type:
        node_config.diskType = options.disk_type
    if options.image_type:
        node_config.imageType = options.image_type
    self.ParseAdvancedMachineFeatures(options, node_config)
    custom_config = self.messages.CustomImageConfig()
    if options.image:
        custom_config.image = options.image
    if options.image_project:
        custom_config.imageProject = options.image_project
    if options.image_family:
        custom_config.imageFamily = options.image_family
    if options.image or options.image_project or options.image_family:
        node_config.nodeImageConfig = custom_config
    NodeIdentityOptionsToNodeConfig(options, node_config)
    if options.local_ssd_count:
        node_config.localSsdCount = options.local_ssd_count
    self._AddLocalSSDVolumeConfigsToNodeConfig(node_config, options)
    self._AddEphemeralStorageToNodeConfig(node_config, options)
    self._AddEphemeralStorageLocalSsdToNodeConfig(node_config, options)
    self._AddLocalNvmeSsdBlockToNodeConfig(node_config, options)
    self._AddStoragePoolsToNodeConfig(node_config, options)
    if options.enable_confidential_storage:
        node_config.enableConfidentialStorage = options.enable_confidential_storage
    if options.boot_disk_kms_key:
        node_config.bootDiskKmsKey = options.boot_disk_kms_key
    if options.tags:
        node_config.tags = options.tags
    else:
        node_config.tags = []
    self.ParseAcceleratorOptions(options, node_config)
    _AddMetadataToNodeConfig(node_config, options)
    _AddLabelsToNodeConfig(node_config, options)
    _AddNodeLabelsToNodeConfig(node_config, options)
    self._AddNodeTaintsToNodeConfig(node_config, options)
    if options.resource_manager_tags is not None:
        tags = options.resource_manager_tags
        node_config.resourceManagerTags = self._ResourceManagerTags(tags)
    if options.preemptible:
        node_config.preemptible = options.preemptible
    if options.spot:
        node_config.spot = options.spot
    if options.min_cpu_platform is not None:
        node_config.minCpuPlatform = options.min_cpu_platform
    if options.node_group is not None:
        node_config.nodeGroup = options.node_group
    if options.enable_gcfs is not None:
        gcfs_config = self.messages.GcfsConfig(enabled=options.enable_gcfs)
        node_config.gcfsConfig = gcfs_config
    if options.enable_image_streaming is not None:
        gcfs_config = self.messages.GcfsConfig(enabled=options.enable_image_streaming)
        node_config.gcfsConfig = gcfs_config
    if options.gvnic is not None:
        gvnic = self.messages.VirtualNIC(enabled=options.gvnic)
        node_config.gvnic = gvnic
    if options.enable_confidential_nodes:
        confidential_nodes = self.messages.ConfidentialNodes(enabled=options.enable_confidential_nodes)
        node_config.confidentialNodes = confidential_nodes
    if options.enable_fast_socket is not None:
        fast_socket = self.messages.FastSocket(enabled=options.enable_fast_socket)
        node_config.fastSocket = fast_socket
    if options.maintenance_interval:
        node_config.stableFleetConfig = _GetStableFleetConfig(options, self.messages)
    if options.logging_variant is not None:
        logging_config = self.messages.NodePoolLoggingConfig()
        logging_config.variantConfig = self.messages.LoggingVariantConfig(variant=VariantConfigEnumFromString(self.messages, options.logging_variant))
        node_config.loggingConfig = logging_config
    if options.host_maintenance_interval:
        node_config.hostMaintenancePolicy = _GetHostMaintenancePolicy(options, self.messages)
    if options.containerd_config_from_file is not None:
        node_config.containerdConfig = self.messages.ContainerdConfig()
        util.LoadContainerdConfigFromYAML(node_config.containerdConfig, options.containerd_config_from_file, self.messages)
    self._AddWorkloadMetadataToNodeConfig(node_config, options, self.messages)
    _AddLinuxNodeConfigToNodeConfig(node_config, options, self.messages)
    _AddShieldedInstanceConfigToNodeConfig(node_config, options, self.messages)
    _AddReservationAffinityToNodeConfig(node_config, options, self.messages)
    _AddSandboxConfigToNodeConfig(node_config, options, self.messages)
    _AddWindowsNodeConfigToNodeConfig(node_config, options, self.messages)
    pool = self.messages.NodePool(name=node_pool_ref.nodePoolId, initialNodeCount=options.num_nodes, config=node_config, version=options.node_version, management=self._GetNodeManagement(options))
    if options.enable_autoscaling or options.enable_autoprovisioning:
        pool.autoscaling = self.messages.NodePoolAutoscaling()
    if options.enable_autoscaling:
        pool.autoscaling.enabled = options.enable_autoscaling
        pool.autoscaling.minNodeCount = options.min_nodes
        pool.autoscaling.maxNodeCount = options.max_nodes
        pool.autoscaling.totalMinNodeCount = options.total_min_nodes
        pool.autoscaling.totalMaxNodeCount = options.total_max_nodes
        if options.location_policy is not None:
            pool.autoscaling.locationPolicy = LocationPolicyEnumFromString(self.messages, options.location_policy)
    if options.enable_best_effort_provision:
        pool.bestEffortProvisioning = self.messages.BestEffortProvisioning()
        pool.bestEffortProvisioning.enabled = True
        pool.bestEffortProvisioning.minProvisionNodes = options.min_provision_nodes
    if options.enable_autoprovisioning:
        pool.autoscaling.autoprovisioned = options.enable_autoprovisioning
    if options.max_pods_per_node is not None:
        pool.maxPodsConstraint = self.messages.MaxPodsConstraint(maxPodsPerNode=options.max_pods_per_node)
    if options.enable_surge_upgrade or options.enable_blue_green_upgrade or options.max_surge_upgrade is not None or (options.max_unavailable_upgrade is not None) or (options.standard_rollout_policy is not None) or (options.autoscaled_rollout_policy is not None) or (options.node_pool_soak_duration is not None):
        pool.upgradeSettings = self.messages.UpgradeSettings()
        pool.upgradeSettings = self.UpdateUpgradeSettings(None, options, pool=pool)
    if options.node_locations is not None:
        pool.locations = sorted(options.node_locations)
    if options.system_config_from_file is not None:
        util.LoadSystemConfigFromYAML(node_config, options.system_config_from_file, options.enable_insecure_kubelet_readonly_port, self.messages)
    if options.enable_insecure_kubelet_readonly_port is not None:
        if node_config.kubeletConfig is None:
            node_config.kubeletConfig = self.messages.NodeKubeletConfig()
        node_config.kubeletConfig.insecureKubeletReadonlyPortEnabled = options.enable_insecure_kubelet_readonly_port
    pool.networkConfig = self._GetNetworkConfig(options)
    if options.network_performance_config:
        pool.networkConfig.networkPerformanceConfig = self._GetNetworkPerformanceConfig(options)
    if options.placement_type == 'COMPACT' or options.placement_policy is not None:
        pool.placementPolicy = self.messages.PlacementPolicy()
    if options.placement_type == 'COMPACT':
        pool.placementPolicy.type = self.messages.PlacementPolicy.TypeValueValuesEnum.COMPACT
    if options.placement_policy is not None:
        pool.placementPolicy.policyName = options.placement_policy
    if options.tpu_topology:
        if pool.placementPolicy is None:
            pool.placementPolicy = self.messages.PlacementPolicy()
        pool.placementPolicy.tpuTopology = options.tpu_topology
    if options.enable_queued_provisioning is not None:
        pool.queuedProvisioning = self.messages.QueuedProvisioning()
        pool.queuedProvisioning.enabled = options.enable_queued_provisioning
    if options.sole_tenant_node_affinity_file is not None:
        node_config.soleTenantConfig = util.LoadSoleTenantConfigFromNodeAffinityYaml(options.sole_tenant_node_affinity_file, self.messages)
    if options.secondary_boot_disks is not None:
        mode_enum = self.messages.SecondaryBootDisk.ModeValueValuesEnum
        mode_map = {'CONTAINER_IMAGE_CACHE': mode_enum.CONTAINER_IMAGE_CACHE}
        node_config.secondaryBootDisks = []
        for disk_config in options.secondary_boot_disks:
            disk_image = disk_config['disk-image']
            mode = None
            if 'mode' in disk_config:
                if disk_config['mode'] in mode_map:
                    mode = mode_map[disk_config['mode']]
                else:
                    mode = mode_enum.MODE_UNSPECIFIED
            node_config.secondaryBootDisks.append(self.messages.SecondaryBootDisk(diskImage=disk_image, mode=mode))
    return pool