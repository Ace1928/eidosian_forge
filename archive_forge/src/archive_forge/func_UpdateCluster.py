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
    if options.enable_cost_allocation is not None:
        update = self.messages.ClusterUpdate(desiredCostManagementConfig=self.messages.CostManagementConfig(enabled=options.enable_cost_allocation))
    if options.release_channel is not None:
        update = self.messages.ClusterUpdate(desiredReleaseChannel=_GetReleaseChannel(options, self.messages))
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