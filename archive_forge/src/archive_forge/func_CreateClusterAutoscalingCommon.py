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
def CreateClusterAutoscalingCommon(self, cluster_ref, options, for_update):
    """Create cluster's autoscaling configuration.

    Args:
      cluster_ref: Cluster reference.
      options: Either CreateClusterOptions or UpdateClusterOptions.
      for_update: Is function executed for update operation.

    Returns:
      Cluster's autoscaling configuration.
    """
    cluster = None
    autoscaling = self.messages.ClusterAutoscaling()
    if cluster_ref:
        cluster = self.GetCluster(cluster_ref)
    if cluster and cluster.autoscaling:
        autoscaling.enableNodeAutoprovisioning = cluster.autoscaling.enableNodeAutoprovisioning
    resource_limits = []
    if options.autoprovisioning_config_file is not None:
        config = yaml.load(options.autoprovisioning_config_file)
        resource_limits = config.get(RESOURCE_LIMITS)
        service_account = config.get(SERVICE_ACCOUNT)
        scopes = config.get(SCOPES)
        max_surge_upgrade = None
        max_unavailable_upgrade = None
        upgrade_settings = config.get(UPGRADE_SETTINGS)
        if upgrade_settings:
            max_surge_upgrade = upgrade_settings.get(MAX_SURGE_UPGRADE)
            max_unavailable_upgrade = upgrade_settings.get(MAX_UNAVAILABLE_UPGRADE)
        management_settings = config.get(NODE_MANAGEMENT)
        enable_autoupgrade = None
        enable_autorepair = None
        if management_settings is not None:
            enable_autoupgrade = management_settings.get(ENABLE_AUTO_UPGRADE)
            enable_autorepair = management_settings.get(ENABLE_AUTO_REPAIR)
        autoprovisioning_locations = config.get(AUTOPROVISIONING_LOCATIONS)
        min_cpu_platform = config.get(MIN_CPU_PLATFORM)
        boot_disk_kms_key = config.get(BOOT_DISK_KMS_KEY)
        disk_type = config.get(DISK_TYPE)
        disk_size_gb = config.get(DISK_SIZE_GB)
        autoprovisioning_image_type = config.get(IMAGE_TYPE)
        shielded_instance_config = config.get(SHIELDED_INSTANCE_CONFIG)
        enable_secure_boot = None
        enable_integrity_monitoring = None
        if shielded_instance_config:
            enable_secure_boot = shielded_instance_config.get(ENABLE_SECURE_BOOT)
            enable_integrity_monitoring = shielded_instance_config.get(ENABLE_INTEGRITY_MONITORING)
    else:
        resource_limits = self.ResourceLimitsFromFlags(options)
        service_account = options.autoprovisioning_service_account
        scopes = options.autoprovisioning_scopes
        autoprovisioning_locations = options.autoprovisioning_locations
        max_surge_upgrade = options.autoprovisioning_max_surge_upgrade
        max_unavailable_upgrade = options.autoprovisioning_max_unavailable_upgrade
        enable_autoupgrade = options.enable_autoprovisioning_autoupgrade
        enable_autorepair = options.enable_autoprovisioning_autorepair
        min_cpu_platform = options.autoprovisioning_min_cpu_platform
        boot_disk_kms_key = None
        disk_type = None
        disk_size_gb = None
        autoprovisioning_image_type = options.autoprovisioning_image_type
        enable_secure_boot = None
        enable_integrity_monitoring = None
    if options.enable_autoprovisioning is not None:
        autoscaling.enableNodeAutoprovisioning = options.enable_autoprovisioning
        if resource_limits is None:
            resource_limits = []
        autoscaling.resourceLimits = resource_limits
        if scopes is None:
            scopes = []
        management = None
        upgrade_settings = None
        if max_surge_upgrade is not None or max_unavailable_upgrade is not None or options.enable_autoprovisioning_blue_green_upgrade or options.enable_autoprovisioning_surge_upgrade or (options.autoprovisioning_standard_rollout_policy is not None) or (options.autoprovisioning_node_pool_soak_duration is not None):
            upgrade_settings = self.UpdateUpgradeSettingsForNAP(options, max_surge_upgrade, max_unavailable_upgrade)
        if enable_autorepair is not None or enable_autorepair is not None:
            management = self.messages.NodeManagement(autoUpgrade=enable_autoupgrade, autoRepair=enable_autorepair)
        shielded_instance_config = None
        if enable_secure_boot is not None or enable_integrity_monitoring is not None:
            shielded_instance_config = self.messages.ShieldedInstanceConfig()
            shielded_instance_config.enableSecureBoot = enable_secure_boot
            shielded_instance_config.enableIntegrityMonitoring = enable_integrity_monitoring
        if for_update:
            autoscaling.autoprovisioningNodePoolDefaults = self.messages.AutoprovisioningNodePoolDefaults(serviceAccount=service_account, oauthScopes=scopes, upgradeSettings=upgrade_settings, management=management, minCpuPlatform=min_cpu_platform, bootDiskKmsKey=boot_disk_kms_key, diskSizeGb=disk_size_gb, diskType=disk_type, imageType=autoprovisioning_image_type, shieldedInstanceConfig=shielded_instance_config)
        else:
            autoscaling.autoprovisioningNodePoolDefaults = self.messages.AutoprovisioningNodePoolDefaults(serviceAccount=service_account, oauthScopes=scopes, upgradeSettings=upgrade_settings, management=management, minCpuPlatform=min_cpu_platform, bootDiskKmsKey=boot_disk_kms_key, diskSizeGb=disk_size_gb, diskType=disk_type, imageType=autoprovisioning_image_type, shieldedInstanceConfig=shielded_instance_config)
        if autoprovisioning_locations:
            autoscaling.autoprovisioningLocations = sorted(autoprovisioning_locations)
    if options.autoscaling_profile is not None:
        autoscaling.autoscalingProfile = self.CreateAutoscalingProfileCommon(options)
    self.ValidateClusterAutoscaling(autoscaling, for_update)
    return autoscaling