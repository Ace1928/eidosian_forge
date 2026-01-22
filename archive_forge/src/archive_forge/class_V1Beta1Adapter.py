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
class V1Beta1Adapter(V1Adapter):
    """APIAdapter for v1beta1."""

    def CreateCluster(self, cluster_ref, options):
        cluster = self.CreateClusterCommon(cluster_ref, options)
        if options.addons:
            if any((v in options.addons for v in CLOUDRUN_ADDONS)):
                if not options.enable_stackdriver_kubernetes and (options.monitoring is not None and SYSTEM not in options.monitoring or (options.logging is not None and SYSTEM not in options.logging)):
                    raise util.Error(CLOUDRUN_STACKDRIVER_KUBERNETES_DISABLED_ERROR_MSG)
                if INGRESS not in options.addons:
                    raise util.Error(CLOUDRUN_INGRESS_KUBERNETES_DISABLED_ERROR_MSG)
                load_balancer_type = _GetCloudRunLoadBalancerType(options, self.messages)
                cluster.addonsConfig.cloudRunConfig = self.messages.CloudRunConfig(disabled=False, loadBalancerType=load_balancer_type)
            if CLOUDBUILD in options.addons:
                if not options.enable_stackdriver_kubernetes and (options.monitoring is not None and SYSTEM not in options.monitoring or (options.logging is not None and SYSTEM not in options.logging)):
                    raise util.Error(CLOUDBUILD_STACKDRIVER_KUBERNETES_DISABLED_ERROR_MSG)
                cluster.addonsConfig.cloudBuildConfig = self.messages.CloudBuildConfig(enabled=True)
            if BACKUPRESTORE in options.addons:
                cluster.addonsConfig.gkeBackupAgentConfig = self.messages.GkeBackupAgentConfig(enabled=True)
            if ISTIO in options.addons:
                istio_auth = self.messages.IstioConfig.AuthValueValuesEnum.AUTH_NONE
                mtls = self.messages.IstioConfig.AuthValueValuesEnum.AUTH_MUTUAL_TLS
                istio_config = options.istio_config
                if istio_config is not None:
                    auth_config = istio_config.get('auth')
                    if auth_config is not None:
                        if auth_config == 'MTLS_STRICT':
                            istio_auth = mtls
                cluster.addonsConfig.istioConfig = self.messages.IstioConfig(disabled=False, auth=istio_auth)
        if options.enable_autoprovisioning is not None or options.autoscaling_profile is not None:
            cluster.autoscaling = self.CreateClusterAutoscalingCommon(None, options, False)
        if options.enable_workload_certificates:
            if not options.workload_pool:
                raise util.Error(PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='workload-pool', opt='enable-workload-certificates'))
            if cluster.workloadCertificates is None:
                cluster.workloadCertificates = self.messages.WorkloadCertificates()
            cluster.workloadCertificates.enableCertificates = options.enable_workload_certificates
        if options.enable_alts:
            if not options.workload_pool:
                raise util.Error(PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='workload-pool', opt='enable-alts'))
            if cluster.workloadAltsConfig is None:
                cluster.workloadAltsConfig = self.messages.WorkloadALTSConfig()
            cluster.workloadAltsConfig.enableAlts = options.enable_alts
        if options.enable_gke_oidc:
            cluster.gkeOidcConfig = self.messages.GkeOidcConfig(enabled=options.enable_gke_oidc)
        if options.enable_identity_service:
            cluster.identityServiceConfig = self.messages.IdentityServiceConfig(enabled=options.enable_identity_service)
        if options.enable_master_global_access is not None:
            if cluster.privateClusterConfig is None:
                cluster.privateClusterConfig = self.messages.PrivateClusterConfig()
            cluster.privateClusterConfig.masterGlobalAccessConfig = self.messages.PrivateClusterMasterGlobalAccessConfig(enabled=options.enable_master_global_access)
        if options.security_group is not None:
            cluster.authenticatorGroupsConfig = self.messages.AuthenticatorGroupsConfig(enabled=True, securityGroup=options.security_group)
        _AddPSCPrivateClustersOptionsToClusterForCreateCluster(cluster, options, self.messages)
        cluster_telemetry_type = self._GetClusterTelemetryType(options, cluster.loggingService, cluster.monitoringService)
        if cluster_telemetry_type is not None:
            cluster.clusterTelemetry = self.messages.ClusterTelemetry()
            cluster.clusterTelemetry.type = cluster_telemetry_type
        if cluster.clusterTelemetry:
            cluster.loggingService = None
            cluster.monitoringService = None
        if options.enable_workload_monitoring_eap:
            cluster.workloadMonitoringEnabledEap = True
        if options.enable_service_externalips is not None:
            if cluster.networkConfig is None:
                cluster.networkConfig = self.messages.NetworkConfig()
            cluster.networkConfig.serviceExternalIpsConfig = self.messages.ServiceExternalIPsConfig(enabled=options.enable_service_externalips)
        if options.identity_provider:
            if options.workload_pool:
                cluster.workloadIdentityConfig.identityProvider = options.identity_provider
            else:
                raise util.Error(PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='workload-pool', opt='identity-provider'))
        if options.datapath_provider is not None:
            if cluster.networkConfig is None:
                cluster.networkConfig = self.messages.NetworkConfig()
            if options.datapath_provider.lower() == 'legacy':
                cluster.networkConfig.datapathProvider = self.messages.NetworkConfig.DatapathProviderValueValuesEnum.LEGACY_DATAPATH
            elif options.datapath_provider.lower() == 'advanced':
                cluster.networkConfig.datapathProvider = self.messages.NetworkConfig.DatapathProviderValueValuesEnum.ADVANCED_DATAPATH
            else:
                raise util.Error(DATAPATH_PROVIDER_ILL_SPECIFIED_ERROR_MSG.format(provider=options.datapath_provider))
        cluster.master = _GetMasterForClusterCreate(options, self.messages)
        cluster.kubernetesObjectsExportConfig = _GetKubernetesObjectsExportConfigForClusterCreate(options, self.messages)
        if options.enable_experimental_vertical_pod_autoscaling is not None:
            cluster.verticalPodAutoscaling = self.messages.VerticalPodAutoscaling(enableExperimentalFeatures=options.enable_experimental_vertical_pod_autoscaling)
            if options.enable_experimental_vertical_pod_autoscaling:
                cluster.verticalPodAutoscaling.enabled = True
        if options.enable_cost_allocation:
            cluster.costManagementConfig = self.messages.CostManagementConfig(enabled=True)
        if options.stack_type is not None:
            cluster.ipAllocationPolicy.stackType = util.GetCreateStackTypeMapper(self.messages).GetEnumForChoice(options.stack_type)
        if options.ipv6_access_type is not None:
            cluster.ipAllocationPolicy.ipv6AccessType = util.GetIpv6AccessTypeMapper(self.messages).GetEnumForChoice(options.ipv6_access_type)
        if options.enable_dns_endpoint is not None:
            if cluster.controlPlaneEndpointsConfig is None:
                cluster.controlPlaneEndpointsConfig = self.messages.ControlPlaneEndpointsConfig()
            dns_endpoint_config = self.messages.DNSEndpointConfig(enabled=options.enable_dns_endpoint)
            cluster.controlPlaneEndpointsConfig.dnsEndpointConfig = dns_endpoint_config
        req = self.messages.CreateClusterRequest(parent=ProjectLocation(cluster_ref.projectId, cluster_ref.zone), cluster=cluster)
        operation = self.client.projects_locations_clusters.Create(req)
        return self.ParseOperation(operation.name, cluster_ref.zone)

    def CreateNodePool(self, node_pool_ref, options):
        pool = self.CreateNodePoolCommon(node_pool_ref, options)
        req = self.messages.CreateNodePoolRequest(nodePool=pool, parent=ProjectLocationCluster(node_pool_ref.projectId, node_pool_ref.zone, node_pool_ref.clusterId))
        operation = self.client.projects_locations_clusters_nodePools.Create(req)
        return self.ParseOperation(operation.name, node_pool_ref.zone)

    def UpdateCluster(self, cluster_ref, options):
        update = self.UpdateClusterCommon(cluster_ref, options)
        if options.workload_pool:
            update = self.messages.ClusterUpdate(desiredWorkloadIdentityConfig=self.messages.WorkloadIdentityConfig(workloadPool=options.workload_pool))
        elif options.identity_provider:
            update = self.messages.ClusterUpdate(desiredWorkloadIdentityConfig=self.messages.WorkloadIdentityConfig(identityProvider=options.identity_provider))
        elif options.disable_workload_identity:
            update = self.messages.ClusterUpdate(desiredWorkloadIdentityConfig=self.messages.WorkloadIdentityConfig(workloadPool=''))
        if options.enable_workload_certificates is not None:
            update = self.messages.ClusterUpdate(desiredWorkloadCertificates=self.messages.WorkloadCertificates(enableCertificates=options.enable_workload_certificates))
        if options.enable_alts is not None:
            update = self.messages.ClusterUpdate(desiredWorkloadAltsConfig=self.messages.WorkloadALTSConfig(enableAlts=options.enable_alts))
        if options.enable_gke_oidc is not None:
            update = self.messages.ClusterUpdate(desiredGkeOidcConfig=self.messages.GkeOidcConfig(enabled=options.enable_gke_oidc))
        if options.enable_identity_service is not None:
            update = self.messages.ClusterUpdate(desiredIdentityServiceConfig=self.messages.IdentityServiceConfig(enabled=options.enable_identity_service))
        if options.enable_stackdriver_kubernetes:
            update = self.messages.ClusterUpdate(desiredClusterTelemetry=self.messages.ClusterTelemetry(type=self.messages.ClusterTelemetry.TypeValueValuesEnum.ENABLED))
        elif options.enable_logging_monitoring_system_only:
            update = self.messages.ClusterUpdate(desiredClusterTelemetry=self.messages.ClusterTelemetry(type=self.messages.ClusterTelemetry.TypeValueValuesEnum.SYSTEM_ONLY))
        elif options.enable_stackdriver_kubernetes is not None:
            update = self.messages.ClusterUpdate(desiredClusterTelemetry=self.messages.ClusterTelemetry(type=self.messages.ClusterTelemetry.TypeValueValuesEnum.DISABLED))
        if options.enable_workload_monitoring_eap is not None:
            update = self.messages.ClusterUpdate(desiredWorkloadMonitoringEapConfig=self.messages.WorkloadMonitoringEapConfig(enabled=options.enable_workload_monitoring_eap))
        if options.enable_experimental_vertical_pod_autoscaling is not None:
            update = self.messages.ClusterUpdate(desiredVerticalPodAutoscaling=self.messages.VerticalPodAutoscaling(enableExperimentalFeatures=options.enable_experimental_vertical_pod_autoscaling))
            if options.enable_experimental_vertical_pod_autoscaling:
                update.desiredVerticalPodAutoscaling.enabled = True
        if options.security_group is not None:
            update = self.messages.ClusterUpdate(desiredAuthenticatorGroupsConfig=self.messages.AuthenticatorGroupsConfig(enabled=True, securityGroup=options.security_group))
        master = _GetMasterForClusterUpdate(options, self.messages)
        if master is not None:
            update = self.messages.ClusterUpdate(desiredMaster=master)
        kubernetes_objects_export_config = _GetKubernetesObjectsExportConfigForClusterUpdate(options, self.messages)
        if kubernetes_objects_export_config is not None:
            update = self.messages.ClusterUpdate(desiredKubernetesObjectsExportConfig=kubernetes_objects_export_config)
        if options.enable_service_externalips is not None:
            update = self.messages.ClusterUpdate(desiredServiceExternalIpsConfig=self.messages.ServiceExternalIPsConfig(enabled=options.enable_service_externalips))
        if options.dataplane_v2:
            update = self.messages.ClusterUpdate(desiredDatapathProvider=self.messages.ClusterUpdate.DesiredDatapathProviderValueValuesEnum.ADVANCED_DATAPATH)
        if options.enable_cost_allocation is not None:
            update = self.messages.ClusterUpdate(desiredCostManagementConfig=self.messages.CostManagementConfig(enabled=options.enable_cost_allocation))
        if options.convert_to_autopilot is not None:
            update = self.messages.ClusterUpdate(desiredAutopilot=self.messages.Autopilot(enabled=True))
        if options.convert_to_standard is not None:
            update = self.messages.ClusterUpdate(desiredAutopilot=self.messages.Autopilot(enabled=False))
        if not update:
            raise util.Error(NOTHING_TO_UPDATE_ERROR_MSG)
        if options.disable_addons is not None:
            if options.disable_addons.get(ISTIO) is not None:
                istio_auth = self.messages.IstioConfig.AuthValueValuesEnum.AUTH_NONE
                mtls = self.messages.IstioConfig.AuthValueValuesEnum.AUTH_MUTUAL_TLS
                istio_config = options.istio_config
                if istio_config is not None:
                    auth_config = istio_config.get('auth')
                    if auth_config is not None:
                        if auth_config == 'MTLS_STRICT':
                            istio_auth = mtls
                update.desiredAddonsConfig.istioConfig = self.messages.IstioConfig(disabled=options.disable_addons.get(ISTIO), auth=istio_auth)
            if any((options.disable_addons.get(v) is not None for v in CLOUDRUN_ADDONS)):
                load_balancer_type = _GetCloudRunLoadBalancerType(options, self.messages)
                update.desiredAddonsConfig.cloudRunConfig = self.messages.CloudRunConfig(disabled=any((options.disable_addons.get(v) or False for v in CLOUDRUN_ADDONS)), loadBalancerType=load_balancer_type)
            if options.disable_addons.get(APPLICATIONMANAGER) is not None:
                update.desiredAddonsConfig.kalmConfig = self.messages.KalmConfig(enabled=not options.disable_addons.get(APPLICATIONMANAGER))
            if options.disable_addons.get(CLOUDBUILD) is not None:
                update.desiredAddonsConfig.cloudBuildConfig = self.messages.CloudBuildConfig(enabled=not options.disable_addons.get(CLOUDBUILD))
        op = self.client.projects_locations_clusters.Update(self.messages.UpdateClusterRequest(name=ProjectLocationCluster(cluster_ref.projectId, cluster_ref.zone, cluster_ref.clusterId), update=update))
        return self.ParseOperation(op.name, cluster_ref.zone)

    def CompleteConvertToAutopilot(self, cluster_ref):
        """Commmit the Autopilot conversion operation.

    Args:
      cluster_ref: cluster resource to commit conversion.

    Returns:
      The operation to be executed.

    Raises:
      exceptions.HttpException: if cluster cannot be found or caller is missing
        permissions. Will attempt to find similar clusters in other zones for a
        more useful error if the user has list permissions.
    """
        try:
            op = self.client.projects_locations_clusters.CompleteConvertToAutopilot(self.messages.ContainerProjectsLocationsClustersCompleteConvertToAutopilotRequest(name=ProjectLocationCluster(cluster_ref.projectId, cluster_ref.zone, cluster_ref.clusterId)))
            return self.ParseOperation(op.name, cluster_ref.zone)
        except apitools_exceptions.HttpNotFoundError as error:
            api_error = exceptions.HttpException(error, util.HTTP_ERROR_FORMAT)
            self.CheckClusterOtherZones(cluster_ref, api_error)
        except apitools_exceptions.HttpError as error:
            raise exceptions.HttpException(error, util.HTTP_ERROR_FORMAT)

    def CreateClusterAutoscalingCommon(self, cluster_ref, options, for_update):
        """Create cluster's autoscaling configuration.

    Args:
      cluster_ref: Cluster reference.
      options: Either CreateClusterOptions or UpdateClusterOptions.
      for_update: Is function executed for update operation.

    Returns:
      Cluster's autoscaling configuration.
    """
        autoscaling = self.messages.ClusterAutoscaling()
        cluster = self.GetCluster(cluster_ref) if cluster_ref else None
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
            if management_settings:
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
            autoscaling.resourceLimits = resource_limits or []
            if scopes is None:
                scopes = []
            management = None
            upgrade_settings = None
            if max_surge_upgrade is not None or max_unavailable_upgrade is not None or options.enable_autoprovisioning_blue_green_upgrade or options.enable_autoprovisioning_surge_upgrade or (options.autoprovisioning_standard_rollout_policy is not None) or (options.autoprovisioning_node_pool_soak_duration is not None):
                upgrade_settings = self.UpdateUpgradeSettingsForNAP(options, max_surge_upgrade, max_unavailable_upgrade)
            if enable_autorepair is not None or enable_autoupgrade is not None:
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

    def ValidateClusterAutoscaling(self, autoscaling, for_update):
        """Validate cluster autoscaling configuration.

    Args:
      autoscaling: autoscaling configuration to be validated.
      for_update: Is function executed for update operation.

    Raises:
      Error if the new configuration is invalid.
    """
        if autoscaling.enableNodeAutoprovisioning:
            if not for_update or autoscaling.resourceLimits:
                cpu_found = any((limit.resourceType == 'cpu' for limit in autoscaling.resourceLimits))
                mem_found = any((limit.resourceType == 'memory' for limit in autoscaling.resourceLimits))
                if not cpu_found or not mem_found:
                    raise util.Error(NO_AUTOPROVISIONING_LIMITS_ERROR_MSG)
                defaults = autoscaling.autoprovisioningNodePoolDefaults
                if defaults:
                    if defaults.upgradeSettings:
                        max_surge_found = defaults.upgradeSettings.maxSurge is not None
                        max_unavailable_found = defaults.upgradeSettings.maxUnavailable is not None
                        if max_unavailable_found != max_surge_found:
                            raise util.Error(BOTH_AUTOPROVISIONING_UPGRADE_SETTINGS_ERROR_MSG)
                    if defaults.management:
                        auto_upgrade_found = defaults.management.autoUpgrade is not None
                        auto_repair_found = defaults.management.autoRepair is not None
                        if auto_repair_found != auto_upgrade_found:
                            raise util.Error(BOTH_AUTOPROVISIONING_MANAGEMENT_SETTINGS_ERROR_MSG)
                    if defaults.shieldedInstanceConfig:
                        secure_boot_found = defaults.shieldedInstanceConfig.enableSecureBoot is not None
                        integrity_monitoring_found = defaults.shieldedInstanceConfig.enableIntegrityMonitoring is not None
                        if secure_boot_found != integrity_monitoring_found:
                            raise util.Error(BOTH_AUTOPROVISIONING_SHIELDED_INSTANCE_SETTINGS_ERROR_MSG)
        elif autoscaling.resourceLimits:
            raise util.Error(LIMITS_WITHOUT_AUTOPROVISIONING_MSG)
        elif autoscaling.autoprovisioningNodePoolDefaults and (autoscaling.autoprovisioningNodePoolDefaults.serviceAccount or autoscaling.autoprovisioningNodePoolDefaults.oauthScopes or autoscaling.autoprovisioningNodePoolDefaults.management or autoscaling.autoprovisioningNodePoolDefaults.upgradeSettings):
            raise util.Error(DEFAULTS_WITHOUT_AUTOPROVISIONING_MSG)

    def UpdateNodePool(self, node_pool_ref, options):
        if options.IsAutoscalingUpdate():
            autoscaling = self.UpdateNodePoolAutoscaling(node_pool_ref, options)
            update = self.messages.ClusterUpdate(desiredNodePoolId=node_pool_ref.nodePoolId, desiredNodePoolAutoscaling=autoscaling)
            operation = self.client.projects_locations_clusters.Update(self.messages.UpdateClusterRequest(name=ProjectLocationCluster(node_pool_ref.projectId, node_pool_ref.zone, node_pool_ref.clusterId), update=update))
            return self.ParseOperation(operation.name, node_pool_ref.zone)
        elif options.IsNodePoolManagementUpdate():
            management = self.UpdateNodePoolNodeManagement(node_pool_ref, options)
            req = self.messages.SetNodePoolManagementRequest(name=ProjectLocationClusterNodePool(node_pool_ref.projectId, node_pool_ref.zone, node_pool_ref.clusterId, node_pool_ref.nodePoolId), management=management)
            operation = self.client.projects_locations_clusters_nodePools.SetManagement(req)
        elif options.IsUpdateNodePoolRequest():
            req = self.UpdateNodePoolRequest(node_pool_ref, options)
            operation = self.client.projects_locations_clusters_nodePools.Update(req)
        else:
            raise util.Error('Unhandled node pool update mode')
        return self.ParseOperation(operation.name, node_pool_ref.zone)