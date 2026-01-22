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
def CreateClusterCommon(self, cluster_ref, options):
    """Returns a CreateCluster operation."""
    node_config = self.ParseNodeConfig(options)
    pools = self.ParseNodePools(options, node_config)
    cluster = self.messages.Cluster(name=cluster_ref.clusterId, nodePools=pools)
    if options.additional_zones:
        cluster.locations = sorted([cluster_ref.zone] + options.additional_zones)
    if options.node_locations:
        cluster.locations = sorted(options.node_locations)
    if options.cluster_version:
        cluster.initialClusterVersion = options.cluster_version
    if options.network:
        cluster.network = options.network
    if options.cluster_ipv4_cidr:
        cluster.clusterIpv4Cidr = options.cluster_ipv4_cidr
    if options.enable_stackdriver_kubernetes is not None:
        if options.enable_stackdriver_kubernetes:
            cluster.loggingService = 'logging.googleapis.com/kubernetes'
            cluster.monitoringService = 'monitoring.googleapis.com/kubernetes'
            if options.enable_cloud_logging is not None and (not options.enable_cloud_logging):
                cluster.loggingService = 'none'
            if options.enable_cloud_monitoring is not None and (not options.enable_cloud_monitoring):
                cluster.monitoringService = 'none'
        else:
            cluster.loggingService = 'none'
            cluster.monitoringService = 'none'
    else:
        if options.enable_cloud_logging is not None:
            if options.enable_cloud_logging:
                cluster.loggingService = 'logging.googleapis.com'
            else:
                cluster.loggingService = 'none'
        if options.enable_cloud_monitoring is not None:
            if options.enable_cloud_monitoring:
                cluster.monitoringService = 'monitoring.googleapis.com'
            else:
                cluster.monitoringService = 'none'
    if options.subnetwork:
        cluster.subnetwork = options.subnetwork
    if options.addons:
        addons = self._AddonsConfig(disable_ingress=INGRESS not in options.addons and (not options.autopilot), disable_hpa=HPA not in options.addons and (not options.autopilot), disable_dashboard=DASHBOARD not in options.addons, disable_network_policy=NETWORK_POLICY not in options.addons, enable_node_local_dns=NODELOCALDNS in options.addons or None, enable_gcepd_csi_driver=GCEPDCSIDRIVER in options.addons, enable_filestore_csi_driver=GCPFILESTORECSIDRIVER in options.addons, enable_application_manager=APPLICATIONMANAGER in options.addons, enable_cloud_build=CLOUDBUILD in options.addons, enable_backup_restore=BACKUPRESTORE in options.addons, enable_gcsfuse_csi_driver=GCSFUSECSIDRIVER in options.addons, enable_stateful_ha=STATEFULHA in options.addons, enable_parallelstore_csi_driver=PARALLELSTORECSIDRIVER in options.addons)
        if CONFIGCONNECTOR in options.addons:
            if not options.enable_stackdriver_kubernetes and (options.monitoring is not None and SYSTEM not in options.monitoring or (options.logging is not None and SYSTEM not in options.logging)):
                raise util.Error(CONFIGCONNECTOR_STACKDRIVER_KUBERNETES_DISABLED_ERROR_MSG)
            if options.workload_pool is None:
                raise util.Error(CONFIGCONNECTOR_WORKLOAD_IDENTITY_DISABLED_ERROR_MSG)
            addons.configConnectorConfig = self.messages.ConfigConnectorConfig(enabled=True)
        cluster.addonsConfig = addons
    self.ParseMasterAuthorizedNetworkOptions(options, cluster)
    if options.enable_kubernetes_alpha:
        cluster.enableKubernetesAlpha = options.enable_kubernetes_alpha
    if options.alpha_cluster_feature_gates:
        if not options.enable_kubernetes_alpha:
            raise util.Error(ALPHA_CLUSTER_FEATURE_GATES_WITHOUT_ENABLE_KUBERNETES_ALPHA_ERROR_MSG)
        cluster.alphaClusterFeatureGates = options.alpha_cluster_feature_gates
    else:
        cluster.alphaClusterFeatureGates = []
    if options.default_max_pods_per_node is not None:
        if not options.enable_ip_alias:
            raise util.Error(DEFAULT_MAX_PODS_PER_NODE_WITHOUT_IP_ALIAS_ERROR_MSG)
        cluster.defaultMaxPodsConstraint = self.messages.MaxPodsConstraint(maxPodsPerNode=options.default_max_pods_per_node)
    if options.disable_default_snat:
        if not options.enable_ip_alias:
            raise util.Error(DISABLE_DEFAULT_SNAT_WITHOUT_IP_ALIAS_ERROR_MSG)
        if not options.enable_private_nodes:
            raise util.Error(DISABLE_DEFAULT_SNAT_WITHOUT_PRIVATE_NODES_ERROR_MSG)
        default_snat_status = self.messages.DefaultSnatStatus(disabled=options.disable_default_snat)
        if cluster.networkConfig is None:
            cluster.networkConfig = self.messages.NetworkConfig(defaultSnatStatus=default_snat_status)
        else:
            cluster.networkConfig.defaultSnatStatus = default_snat_status
    if options.dataplane_v2 is not None and options.dataplane_v2:
        if cluster.networkConfig is None:
            cluster.networkConfig = self.messages.NetworkConfig()
        cluster.networkConfig.datapathProvider = self.messages.NetworkConfig.DatapathProviderValueValuesEnum.ADVANCED_DATAPATH
    if options.enable_l4_ilb_subsetting:
        if cluster.networkConfig is None:
            cluster.networkConfig = self.messages.NetworkConfig(enableL4ilbSubsetting=options.enable_l4_ilb_subsetting)
        else:
            cluster.networkConfig.enableL4ilbSubsetting = options.enable_l4_ilb_subsetting
    dns_config = self.ParseClusterDNSOptions(options)
    if dns_config is not None:
        if cluster.networkConfig is None:
            cluster.networkConfig = self.messages.NetworkConfig(dnsConfig=dns_config)
        else:
            cluster.networkConfig.dnsConfig = dns_config
    gateway_config = self.ParseGatewayOptions(options)
    if gateway_config is not None:
        if cluster.networkConfig is None:
            cluster.networkConfig = self.messages.NetworkConfig(gatewayApiConfig=gateway_config)
        else:
            cluster.networkConfig.gatewayApiConfig = gateway_config
    if options.enable_legacy_authorization is not None:
        cluster.legacyAbac = self.messages.LegacyAbac(enabled=bool(options.enable_legacy_authorization))
    if options.enable_network_policy:
        cluster.networkPolicy = self.messages.NetworkPolicy(enabled=options.enable_network_policy, provider=self.messages.NetworkPolicy.ProviderValueValuesEnum.CALICO)
    if options.enable_binauthz is not None:
        cluster.binaryAuthorization = self.messages.BinaryAuthorization(enabled=options.enable_binauthz)
    if options.binauthz_evaluation_mode is not None:
        if options.binauthz_policy_bindings is not None:
            cluster.binaryAuthorization = self.messages.BinaryAuthorization(evaluationMode=util.GetBinauthzEvaluationModeMapper(self.messages, hidden=False).GetEnumForChoice(options.binauthz_evaluation_mode))
            for binding in options.binauthz_policy_bindings:
                cluster.binaryAuthorization.policyBindings.append(self.messages.PolicyBinding(name=binding['name']))
        else:
            cluster.binaryAuthorization = self.messages.BinaryAuthorization(evaluationMode=util.GetBinauthzEvaluationModeMapper(self.messages, hidden=False).GetEnumForChoice(options.binauthz_evaluation_mode))
    if options.binauthz_policy_bindings and (not options.binauthz_evaluation_mode):
        raise util.Error(PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='binauthz-evaluation-mode', opt='binauthz-policy-bindings'))
    if options.maintenance_window is not None:
        cluster.maintenancePolicy = self.messages.MaintenancePolicy(window=self.messages.MaintenanceWindow(dailyMaintenanceWindow=self.messages.DailyMaintenanceWindow(startTime=options.maintenance_window)))
    elif options.maintenance_window_start is not None:
        window_start = options.maintenance_window_start.isoformat()
        window_end = options.maintenance_window_end.isoformat()
        cluster.maintenancePolicy = self.messages.MaintenancePolicy(window=self.messages.MaintenanceWindow(recurringWindow=self.messages.RecurringTimeWindow(window=self.messages.TimeWindow(startTime=window_start, endTime=window_end), recurrence=options.maintenance_window_recurrence)))
    self.ParseResourceLabels(options, cluster)
    if options.enable_pod_security_policy is not None:
        cluster.podSecurityPolicyConfig = self.messages.PodSecurityPolicyConfig(enabled=options.enable_pod_security_policy)
    if options.security_group is not None:
        cluster.authenticatorGroupsConfig = self.messages.AuthenticatorGroupsConfig(enabled=True, securityGroup=options.security_group)
    if options.enable_shielded_nodes is not None:
        cluster.shieldedNodes = self.messages.ShieldedNodes(enabled=options.enable_shielded_nodes)
    if options.workload_pool:
        cluster.workloadIdentityConfig = self.messages.WorkloadIdentityConfig(workloadPool=options.workload_pool)
    self.ParseIPAliasOptions(options, cluster)
    self.ParseAllowRouteOverlapOptions(options, cluster)
    self.ParsePrivateClusterOptions(options, cluster)
    self.ParseTpuOptions(options, cluster)
    if options.enable_vertical_pod_autoscaling is not None:
        cluster.verticalPodAutoscaling = self.messages.VerticalPodAutoscaling(enabled=options.enable_vertical_pod_autoscaling)
    if options.resource_usage_bigquery_dataset:
        bigquery_destination = self.messages.BigQueryDestination(datasetId=options.resource_usage_bigquery_dataset)
        cluster.resourceUsageExportConfig = self.messages.ResourceUsageExportConfig(bigqueryDestination=bigquery_destination)
        if options.enable_network_egress_metering:
            cluster.resourceUsageExportConfig.enableNetworkEgressMetering = True
        if options.enable_resource_consumption_metering is not None:
            cluster.resourceUsageExportConfig.consumptionMeteringConfig = self.messages.ConsumptionMeteringConfig(enabled=options.enable_resource_consumption_metering)
    elif options.enable_network_egress_metering is not None:
        raise util.Error(ENABLE_NETWORK_EGRESS_METERING_ERROR_MSG)
    elif options.enable_resource_consumption_metering is not None:
        raise util.Error(ENABLE_RESOURCE_CONSUMPTION_METERING_ERROR_MSG)
    if options.user is not None or options.issue_client_certificate is not None:
        cluster.masterAuth = self.messages.MasterAuth(username=options.user, password=options.password)
        if options.issue_client_certificate is not None:
            cluster.masterAuth.clientCertificateConfig = self.messages.ClientCertificateConfig(issueClientCertificate=options.issue_client_certificate)
    if options.enable_intra_node_visibility is not None:
        if cluster.networkConfig is None:
            cluster.networkConfig = self.messages.NetworkConfig(enableIntraNodeVisibility=options.enable_intra_node_visibility)
        else:
            cluster.networkConfig.enableIntraNodeVisibility = options.enable_intra_node_visibility
    if options.database_encryption_key:
        cluster.databaseEncryption = self.messages.DatabaseEncryption(keyName=options.database_encryption_key, state=self.messages.DatabaseEncryption.StateValueValuesEnum.ENCRYPTED)
    if options.boot_disk_kms_key:
        for pool in cluster.nodePools:
            pool.config.bootDiskKmsKey = options.boot_disk_kms_key
    cluster.releaseChannel = _GetReleaseChannel(options, self.messages)
    if options.autopilot:
        cluster.autopilot = self.messages.Autopilot()
        cluster.autopilot.enabled = True
        if options.workload_policies:
            if cluster.autopilot.workloadPolicyConfig is None:
                cluster.autopilot.workloadPolicyConfig = self.messages.WorkloadPolicyConfig()
            if options.workload_policies == 'allow-net-admin':
                cluster.autopilot.workloadPolicyConfig.allowNetAdmin = True
        if options.enable_secret_manager:
            if cluster.secretManagerConfig is None:
                cluster.secretManagerConfig = self.messages.SecretManagerConfig(enabled=False)
        if options.boot_disk_kms_key:
            if cluster.autoscaling is None:
                cluster.autoscaling = self.messages.ClusterAutoscaling()
            if cluster.autoscaling.autoprovisioningNodePoolDefaults is None:
                cluster.autoscaling.autoprovisioningNodePoolDefaults = self.messages.AutoprovisioningNodePoolDefaults()
            cluster.autoscaling.autoprovisioningNodePoolDefaults.bootDiskKmsKey = options.boot_disk_kms_key
    if options.enable_confidential_nodes:
        cluster.confidentialNodes = self.messages.ConfidentialNodes(enabled=options.enable_confidential_nodes)
    if options.private_ipv6_google_access_type is not None:
        if cluster.networkConfig is None:
            cluster.networkConfig = self.messages.NetworkConfig()
        cluster.networkConfig.privateIpv6GoogleAccess = util.GetPrivateIpv6GoogleAccessTypeMapper(self.messages, hidden=False).GetEnumForChoice(options.private_ipv6_google_access_type)
    if options.enable_insecure_kubelet_readonly_port is not None:
        if options.autopilot:
            raise util.Error(NODECONFIGDEFAULTS_READONLY_PORT_NOT_SUPPORTED)
        if cluster.nodePoolDefaults is None:
            cluster.nodePoolDefaults = self.messages.NodePoolDefaults()
        if cluster.nodePoolDefaults.nodeConfigDefaults is None:
            cluster.nodePoolDefaults.nodeConfigDefaults = self.messages.NodeConfigDefaults()
        if cluster.nodePoolDefaults.nodeConfigDefaults.nodeKubeletConfig is None:
            cluster.nodePoolDefaults.nodeConfigDefaults.nodeKubeletConfig = self.messages.NodeKubeletConfig()
        cluster.nodePoolDefaults.nodeConfigDefaults.nodeKubeletConfig.insecureKubeletReadonlyPortEnabled = options.enable_insecure_kubelet_readonly_port
    if options.autoprovisioning_enable_insecure_kubelet_readonly_port is not None:
        if cluster.nodePoolAutoConfig is None:
            cluster.nodePoolAutoConfig = self.messages.NodePoolAutoConfig()
        if cluster.nodePoolAutoConfig.nodeKubeletConfig is None:
            cluster.nodePoolAutoConfig.nodeKubeletConfig = self.messages.NodeKubeletConfig()
        cluster.nodePoolAutoConfig.nodeKubeletConfig.insecureKubeletReadonlyPortEnabled = options.autoprovisioning_enable_insecure_kubelet_readonly_port
    if options.enable_gcfs:
        if cluster.nodePoolDefaults is None:
            cluster.nodePoolDefaults = self.messages.NodePoolDefaults()
        if cluster.nodePoolDefaults.nodeConfigDefaults is None:
            cluster.nodePoolDefaults.nodeConfigDefaults = self.messages.NodeConfigDefaults()
        cluster.nodePoolDefaults.nodeConfigDefaults.gcfsConfig = self.messages.GcfsConfig(enabled=options.enable_gcfs)
    if options.containerd_config_from_file is not None:
        if cluster.nodePoolDefaults is None:
            cluster.nodePoolDefaults = self.messages.NodePoolDefaults()
        if cluster.nodePoolDefaults.nodeConfigDefaults is None:
            cluster.nodePoolDefaults.nodeConfigDefaults = self.messages.NodeConfigDefaults()
        cluster.nodePoolDefaults.nodeConfigDefaults.containerdConfig = self.messages.ContainerdConfig()
        util.LoadContainerdConfigFromYAML(cluster.nodePoolDefaults.nodeConfigDefaults.containerdConfig, options.containerd_config_from_file, self.messages)
    if options.autoprovisioning_network_tags:
        if cluster.nodePoolAutoConfig is None:
            cluster.nodePoolAutoConfig = self.messages.NodePoolAutoConfig()
        cluster.nodePoolAutoConfig.networkTags = self.messages.NetworkTags(tags=options.autoprovisioning_network_tags)
    if options.autoprovisioning_resource_manager_tags is not None:
        if cluster.nodePoolAutoConfig is None:
            cluster.nodePoolAutoConfig = self.messages.NodePoolAutoConfig()
        rm_tags = self._ResourceManagerTags(options.autoprovisioning_resource_manager_tags)
        cluster.nodePoolAutoConfig.resourceManagerTags = rm_tags
    if options.enable_image_streaming:
        if cluster.nodePoolDefaults is None:
            cluster.nodePoolDefaults = self.messages.NodePoolDefaults()
        if cluster.nodePoolDefaults.nodeConfigDefaults is None:
            cluster.nodePoolDefaults.nodeConfigDefaults = self.messages.NodeConfigDefaults()
        cluster.nodePoolDefaults.nodeConfigDefaults.gcfsConfig = self.messages.GcfsConfig(enabled=options.enable_image_streaming)
    if options.maintenance_interval:
        if cluster.nodePoolDefaults is None:
            cluster.nodePoolDefaults = self.messages.NodePoolDefaults()
        if cluster.nodePoolDefaults.nodeConfigDefaults is None:
            cluster.nodePoolDefaults.nodeConfigDefaults = self.messages.NodeConfigDefaults()
        cluster.nodePoolDefaults.nodeConfigDefaults.stableFleetConfig = _GetStableFleetConfig(options, self.messages)
    if options.enable_mesh_certificates:
        if not options.workload_pool:
            raise util.Error(PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='workload-pool', opt='enable-mesh-certificates'))
        if cluster.meshCertificates is None:
            cluster.meshCertificates = self.messages.MeshCertificates()
        cluster.meshCertificates.enableCertificates = options.enable_mesh_certificates
    _AddNotificationConfigToCluster(cluster, options, self.messages)
    cluster.loggingConfig = _GetLoggingConfig(options, self.messages)
    cluster.monitoringConfig = _GetMonitoringConfig(options, self.messages)
    if options.enable_service_externalips is not None:
        if cluster.networkConfig is None:
            cluster.networkConfig = self.messages.NetworkConfig()
        cluster.networkConfig.serviceExternalIpsConfig = self.messages.ServiceExternalIPsConfig(enabled=options.enable_service_externalips)
    if options.enable_identity_service:
        cluster.identityServiceConfig = self.messages.IdentityServiceConfig(enabled=options.enable_identity_service)
    if options.enable_workload_config_audit is not None:
        if cluster.protectConfig is None:
            cluster.protectConfig = self.messages.ProtectConfig(workloadConfig=self.messages.WorkloadConfig())
        if options.enable_workload_config_audit:
            cluster.protectConfig.workloadConfig.auditMode = self.messages.WorkloadConfig.AuditModeValueValuesEnum.BASIC
        else:
            cluster.protectConfig.workloadConfig.auditMode = self.messages.WorkloadConfig.AuditModeValueValuesEnum.DISABLED
    if options.enable_workload_vulnerability_scanning is not None:
        if cluster.protectConfig is None:
            cluster.protectConfig = self.messages.ProtectConfig()
        if options.enable_workload_vulnerability_scanning:
            cluster.protectConfig.workloadVulnerabilityMode = self.messages.ProtectConfig.WorkloadVulnerabilityModeValueValuesEnum.BASIC
        else:
            cluster.protectConfig.workloadVulnerabilityMode = self.messages.ProtectConfig.WorkloadVulnerabilityModeValueValuesEnum.DISABLED
    if options.pod_autoscaling_direct_metrics_opt_in is not None:
        pod_autoscaling_config = self.messages.PodAutoscaling(directMetricsOptIn=options.pod_autoscaling_direct_metrics_opt_in)
        cluster.podAutoscaling = pod_autoscaling_config
    if options.private_endpoint_subnetwork is not None:
        if cluster.privateClusterConfig is None:
            cluster.privateClusterConfig = self.messages.PrivateClusterConfig()
        cluster.privateClusterConfig.privateEndpointSubnetwork = options.private_endpoint_subnetwork
    if options.managed_config is not None:
        if options.managed_config.lower() == 'autofleet':
            cluster.managedConfig = self.messages.ManagedConfig(type=self.messages.ManagedConfig.TypeValueValuesEnum.AUTOFLEET)
        elif options.managed_config.lower() == 'disabled':
            cluster.managedConfig = self.messages.ManagedConfig(type=self.messages.ManagedConfig.TypeValueValuesEnum.DISABLED)
        else:
            raise util.Error(MANGED_CONFIG_TYPE_NOT_SUPPORTED.format(type=options.managed_config))
    if options.enable_fleet:
        if cluster.fleet is None:
            cluster.fleet = self.messages.Fleet()
        cluster.fleet.project = cluster_ref.projectId
    if options.fleet_project:
        if cluster.fleet is None:
            cluster.fleet = self.messages.Fleet()
        cluster.fleet.project = options.fleet_project
    if options.logging_variant is not None:
        if cluster.nodePoolDefaults is None:
            cluster.nodePoolDefaults = self.messages.NodePoolDefaults()
        if cluster.nodePoolDefaults.nodeConfigDefaults is None:
            cluster.nodePoolDefaults.nodeConfigDefaults = self.messages.NodeConfigDefaults()
        if cluster.nodePoolDefaults.nodeConfigDefaults.loggingConfig is None:
            cluster.nodePoolDefaults.nodeConfigDefaults.loggingConfig = self.messages.NodePoolLoggingConfig()
        cluster.nodePoolDefaults.nodeConfigDefaults.loggingConfig.variantConfig = self.messages.LoggingVariantConfig(variant=VariantConfigEnumFromString(self.messages, options.logging_variant))
    if options.enable_cost_allocation:
        cluster.costManagementConfig = self.messages.CostManagementConfig(enabled=True)
    if options.enable_multi_networking:
        if cluster.networkConfig is None:
            cluster.networkConfig = self.messages.NetworkConfig(enableMultiNetworking=options.enable_multi_networking)
        else:
            cluster.networkConfig.enableMultiNetworking = options.enable_multi_networking
    if options.enable_security_posture is not None:
        if cluster.securityPostureConfig is None:
            cluster.securityPostureConfig = self.messages.SecurityPostureConfig()
        if options.enable_security_posture:
            cluster.securityPostureConfig.mode = self.messages.SecurityPostureConfig.ModeValueValuesEnum.BASIC
        else:
            cluster.securityPostureConfig.mode = self.messages.SecurityPostureConfig.ModeValueValuesEnum.DISABLED
    if options.security_posture is not None:
        if cluster.securityPostureConfig is None:
            cluster.securityPostureConfig = self.messages.SecurityPostureConfig()
        if options.security_posture.lower() == 'enterprise':
            cluster.securityPostureConfig.mode = self.messages.SecurityPostureConfig.ModeValueValuesEnum.ENTERPRISE
        elif options.security_posture.lower() == 'standard':
            cluster.securityPostureConfig.mode = self.messages.SecurityPostureConfig.ModeValueValuesEnum.BASIC
        elif options.security_posture.lower() == 'disabled':
            cluster.securityPostureConfig.mode = self.messages.SecurityPostureConfig.ModeValueValuesEnum.DISABLED
        else:
            raise util.Error(SECURITY_POSTURE_MODE_NOT_SUPPORTED.format(mode=options.security_posture.lower()))
    if options.workload_vulnerability_scanning is not None:
        if cluster.securityPostureConfig is None:
            cluster.securityPostureConfig = self.messages.SecurityPostureConfig()
        if options.workload_vulnerability_scanning.lower() == 'standard':
            cluster.securityPostureConfig.vulnerabilityMode = self.messages.SecurityPostureConfig.VulnerabilityModeValueValuesEnum.VULNERABILITY_BASIC
        elif options.workload_vulnerability_scanning.lower() == 'disabled':
            cluster.securityPostureConfig.vulnerabilityMode = self.messages.SecurityPostureConfig.VulnerabilityModeValueValuesEnum.VULNERABILITY_DISABLED
        elif options.workload_vulnerability_scanning.lower() == 'enterprise':
            cluster.securityPostureConfig.vulnerabilityMode = self.messages.SecurityPostureConfig.VulnerabilityModeValueValuesEnum.VULNERABILITY_ENTERPRISE
        else:
            raise util.Error(WORKLOAD_VULNERABILITY_SCANNING_MODE_NOT_SUPPORTED.format(mode=options.workload_vulnerability_scanning.lower()))
    if options.enable_runtime_vulnerability_insight is not None:
        if cluster.runtimeVulnerabilityInsightConfig is None:
            cluster.runtimeVulnerabilityInsightConfig = self.messages.RuntimeVulnerabilityInsightConfig()
        if options.enable_runtime_vulnerability_insight:
            cluster.runtimeVulnerabilityInsightConfig.mode = self.messages.RuntimeVulnerabilityInsightConfig.ModeValueValuesEnum.PREMIUM_VULNERABILITY_SCAN
        else:
            cluster.runtimeVulnerabilityInsightConfig.mode = self.messages.RuntimeVulnerabilityInsightConfig.ModeValueValuesEnum.DISABLED
    if options.network_performance_config:
        perf = self._GetClusterNetworkPerformanceConfig(options)
        if cluster.networkConfig is None:
            cluster.networkConfig = self.messages.NetworkConfig(networkPerformanceConfig=perf)
        else:
            cluster.networkConfig.networkPerformanceConfig = perf
    if options.enable_k8s_beta_apis:
        cluster.enableK8sBetaApis = self.messages.K8sBetaAPIConfig()
        cluster.enableK8sBetaApis.enabledApis = options.enable_k8s_beta_apis
    if options.host_maintenance_interval:
        if cluster.nodePoolDefaults is None:
            cluster.nodePoolDefaults = self.messages.NodePoolDefaults()
        if cluster.nodePoolDefaults.nodeConfigDefaults is None:
            cluster.nodePoolDefaults.nodeConfigDefaults = self.messages.NodeConfigDefaults()
        cluster.nodePoolDefaults.nodeConfigDefaults.hostMaintenancePolicy = _GetHostMaintenancePolicy(options, self.messages)
    if options.in_transit_encryption is not None:
        if cluster.networkConfig is None:
            cluster.networkConfig = self.messages.NetworkConfig()
        cluster.networkConfig.inTransitEncryptionConfig = util.GetCreateInTransitEncryptionConfigMapper(self.messages).GetEnumForChoice(options.in_transit_encryption)
    if options.enable_secret_manager is not None:
        if cluster.secretManagerConfig is None:
            cluster.secretManagerConfig = self.messages.SecretManagerConfig()
        cluster.secretManagerConfig.enabled = options.enable_secret_manager
    if options.enable_cilium_clusterwide_network_policy is not None:
        if cluster.networkConfig is None:
            cluster.networkConfig = self.messages.NetworkConfig()
        cluster.networkConfig.enableCiliumClusterwideNetworkPolicy = options.enable_cilium_clusterwide_network_policy
    if options.enable_fqdn_network_policy is not None:
        if cluster.networkConfig is None:
            cluster.networkConfig = self.messages.NetworkConfig()
        cluster.networkConfig.enableFqdnNetworkPolicy = options.enable_fqdn_network_policy
    return cluster