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
class UpdateClusterOptions(object):
    """Options to pass to UpdateCluster."""

    def __init__(self, version=None, update_master=None, update_nodes=None, node_pool=None, monitoring_service=None, logging_service=None, enable_stackdriver_kubernetes=None, enable_logging_monitoring_system_only=None, enable_workload_monitoring_eap=None, master_logs=None, no_master_logs=None, enable_master_metrics=None, logging=None, monitoring=None, disable_addons=None, istio_config=None, cloud_run_config=None, cluster_dns=None, cluster_dns_scope=None, cluster_dns_domain=None, disable_additive_vpc_scope=None, additive_vpc_scope_dns_domain=None, enable_autoscaling=None, min_nodes=None, max_nodes=None, total_min_nodes=None, total_max_nodes=None, location_policy=None, image_type=None, image=None, image_project=None, locations=None, enable_master_authorized_networks=None, master_authorized_networks=None, enable_pod_security_policy=None, enable_vertical_pod_autoscaling=None, enable_experimental_vertical_pod_autoscaling=None, enable_intra_node_visibility=None, enable_l4_ilb_subsetting=None, security_profile=None, security_profile_runtime_rules=None, autoscaling_profile=None, enable_peering_route_sharing=None, workload_pool=None, identity_provider=None, disable_workload_identity=None, enable_workload_certificates=None, enable_mesh_certificates=None, enable_alts=None, enable_gke_oidc=None, enable_identity_service=None, enable_shielded_nodes=None, disable_default_snat=None, resource_usage_bigquery_dataset=None, enable_network_egress_metering=None, enable_resource_consumption_metering=None, database_encryption_key=None, disable_database_encryption=None, enable_cost_allocation=None, enable_autoprovisioning=None, autoprovisioning_config_file=None, autoprovisioning_service_account=None, autoprovisioning_scopes=None, autoprovisioning_locations=None, min_cpu=None, max_cpu=None, min_memory=None, max_memory=None, min_accelerator=None, max_accelerator=None, release_channel=None, autoprovisioning_image_type=None, autoprovisioning_max_surge_upgrade=None, autoprovisioning_max_unavailable_upgrade=None, enable_autoprovisioning_autoupgrade=None, enable_autoprovisioning_autorepair=None, autoprovisioning_min_cpu_platform=None, enable_tpu=None, tpu_ipv4_cidr=None, enable_master_global_access=None, enable_tpu_service_networking=None, notification_config=None, private_ipv6_google_access_type=None, kubernetes_objects_changes_target=None, kubernetes_objects_snapshots_target=None, disable_autopilot=None, add_cross_connect_subnetworks=None, remove_cross_connect_subnetworks=None, clear_cross_connect_subnetworks=None, enable_service_externalips=None, security_group=None, enable_gcfs=None, autoprovisioning_network_tags=None, enable_image_streaming=None, enable_managed_prometheus=None, disable_managed_prometheus=None, maintenance_interval=None, dataplane_v2=None, enable_dataplane_v2_metrics=None, disable_dataplane_v2_metrics=None, enable_dataplane_v2_flow_observability=None, disable_dataplane_v2_flow_observability=None, dataplane_v2_observability_mode=None, enable_workload_config_audit=None, pod_autoscaling_direct_metrics_opt_in=None, enable_workload_vulnerability_scanning=None, enable_autoprovisioning_surge_upgrade=None, enable_autoprovisioning_blue_green_upgrade=None, autoprovisioning_standard_rollout_policy=None, autoprovisioning_node_pool_soak_duration=None, enable_private_endpoint=None, enable_google_cloud_access=None, stack_type=None, gateway_api=None, logging_variant=None, additional_pod_ipv4_ranges=None, removed_additional_pod_ipv4_ranges=None, fleet_project=None, enable_fleet=None, clear_fleet_project=None, enable_security_posture=None, network_performance_config=None, enable_k8s_beta_apis=None, security_posture=None, workload_vulnerability_scanning=None, enable_runtime_vulnerability_insight=None, workload_policies=None, remove_workload_policies=None, enable_fqdn_network_policy=None, host_maintenance_interval=None, in_transit_encryption=None, enable_multi_networking=None, containerd_config_from_file=None, autoprovisioning_resource_manager_tags=None, convert_to_autopilot=None, convert_to_standard=None, enable_secret_manager=None, enable_cilium_clusterwide_network_policy=None, enable_insecure_kubelet_readonly_port=None, autoprovisioning_enable_insecure_kubelet_readonly_port=None):
        self.version = version
        self.update_master = bool(update_master)
        self.update_nodes = bool(update_nodes)
        self.node_pool = node_pool
        self.monitoring_service = monitoring_service
        self.logging_service = logging_service
        self.enable_stackdriver_kubernetes = enable_stackdriver_kubernetes
        self.enable_logging_monitoring_system_only = enable_logging_monitoring_system_only
        self.enable_workload_monitoring_eap = enable_workload_monitoring_eap
        self.no_master_logs = no_master_logs
        self.master_logs = master_logs
        self.enable_master_metrics = enable_master_metrics
        self.logging = logging
        self.monitoring = monitoring
        self.disable_addons = disable_addons
        self.istio_config = istio_config
        self.cloud_run_config = cloud_run_config
        self.cluster_dns = cluster_dns
        self.cluster_dns_scope = cluster_dns_scope
        self.cluster_dns_domain = cluster_dns_domain
        self.disable_additive_vpc_scope = disable_additive_vpc_scope
        self.additive_vpc_scope_dns_domain = additive_vpc_scope_dns_domain
        self.enable_autoscaling = enable_autoscaling
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.total_min_nodes = total_min_nodes
        self.total_max_nodes = total_max_nodes
        self.location_policy = location_policy
        self.image_type = image_type
        self.image = image
        self.image_project = image_project
        self.locations = locations
        self.enable_master_authorized_networks = enable_master_authorized_networks
        self.master_authorized_networks = master_authorized_networks
        self.enable_pod_security_policy = enable_pod_security_policy
        self.enable_vertical_pod_autoscaling = enable_vertical_pod_autoscaling
        self.enable_experimental_vertical_pod_autoscaling = enable_experimental_vertical_pod_autoscaling
        self.security_profile = security_profile
        self.security_profile_runtime_rules = security_profile_runtime_rules
        self.autoscaling_profile = autoscaling_profile
        self.enable_intra_node_visibility = enable_intra_node_visibility
        self.enable_l4_ilb_subsetting = enable_l4_ilb_subsetting
        self.enable_peering_route_sharing = enable_peering_route_sharing
        self.workload_pool = workload_pool
        self.identity_provider = identity_provider
        self.disable_workload_identity = disable_workload_identity
        self.enable_workload_certificates = enable_workload_certificates
        self.enable_mesh_certificates = enable_mesh_certificates
        self.enable_alts = enable_alts
        self.enable_gke_oidc = enable_gke_oidc
        self.enable_identity_service = enable_identity_service
        self.enable_shielded_nodes = enable_shielded_nodes
        self.disable_default_snat = disable_default_snat
        self.resource_usage_bigquery_dataset = resource_usage_bigquery_dataset
        self.enable_network_egress_metering = enable_network_egress_metering
        self.enable_resource_consumption_metering = enable_resource_consumption_metering
        self.database_encryption_key = database_encryption_key
        self.disable_database_encryption = disable_database_encryption
        self.enable_cost_allocation = enable_cost_allocation
        self.enable_autoprovisioning = enable_autoprovisioning
        self.autoprovisioning_config_file = autoprovisioning_config_file
        self.autoprovisioning_service_account = autoprovisioning_service_account
        self.autoprovisioning_scopes = autoprovisioning_scopes
        self.autoprovisioning_locations = autoprovisioning_locations
        self.min_cpu = min_cpu
        self.max_cpu = max_cpu
        self.min_memory = min_memory
        self.max_memory = max_memory
        self.min_accelerator = min_accelerator
        self.max_accelerator = max_accelerator
        self.release_channel = release_channel
        self.autoprovisioning_image_type = autoprovisioning_image_type
        self.autoprovisioning_max_surge_upgrade = autoprovisioning_max_surge_upgrade
        self.autoprovisioning_max_unavailable_upgrade = autoprovisioning_max_unavailable_upgrade
        self.enable_autoprovisioning_autoupgrade = enable_autoprovisioning_autoupgrade
        self.enable_autoprovisioning_autorepair = enable_autoprovisioning_autorepair
        self.autoprovisioning_min_cpu_platform = autoprovisioning_min_cpu_platform
        self.enable_tpu = enable_tpu
        self.tpu_ipv4_cidr = tpu_ipv4_cidr
        self.enable_tpu_service_networking = enable_tpu_service_networking
        self.enable_master_global_access = enable_master_global_access
        self.notification_config = notification_config
        self.private_ipv6_google_access_type = private_ipv6_google_access_type
        self.kubernetes_objects_changes_target = kubernetes_objects_changes_target
        self.kubernetes_objects_snapshots_target = kubernetes_objects_snapshots_target
        self.disable_autopilot = disable_autopilot
        self.add_cross_connect_subnetworks = add_cross_connect_subnetworks
        self.remove_cross_connect_subnetworks = remove_cross_connect_subnetworks
        self.clear_cross_connect_subnetworks = clear_cross_connect_subnetworks
        self.enable_service_externalips = enable_service_externalips
        self.security_group = security_group
        self.enable_gcfs = enable_gcfs
        self.autoprovisioning_network_tags = autoprovisioning_network_tags
        self.enable_image_streaming = enable_image_streaming
        self.enable_managed_prometheus = enable_managed_prometheus
        self.disable_managed_prometheus = disable_managed_prometheus
        self.maintenance_interval = maintenance_interval
        self.dataplane_v2 = dataplane_v2
        self.enable_dataplane_v2_metrics = enable_dataplane_v2_metrics
        self.disable_dataplane_v2_metrics = disable_dataplane_v2_metrics
        self.enable_dataplane_v2_flow_observability = enable_dataplane_v2_flow_observability
        self.disable_dataplane_v2_flow_observability = disable_dataplane_v2_flow_observability
        self.dataplane_v2_observability_mode = dataplane_v2_observability_mode
        self.enable_workload_config_audit = enable_workload_config_audit
        self.pod_autoscaling_direct_metrics_opt_in = pod_autoscaling_direct_metrics_opt_in
        self.enable_workload_vulnerability_scanning = enable_workload_vulnerability_scanning
        self.enable_autoprovisioning_surge_upgrade = enable_autoprovisioning_surge_upgrade
        self.enable_autoprovisioning_blue_green_upgrade = enable_autoprovisioning_blue_green_upgrade
        self.autoprovisioning_standard_rollout_policy = autoprovisioning_standard_rollout_policy
        self.autoprovisioning_node_pool_soak_duration = autoprovisioning_node_pool_soak_duration
        self.enable_private_endpoint = enable_private_endpoint
        self.enable_google_cloud_access = enable_google_cloud_access
        self.stack_type = stack_type
        self.gateway_api = gateway_api
        self.logging_variant = logging_variant
        self.additional_pod_ipv4_ranges = additional_pod_ipv4_ranges
        self.removed_additional_pod_ipv4_ranges = removed_additional_pod_ipv4_ranges
        self.fleet_project = fleet_project
        self.enable_fleet = enable_fleet
        self.clear_fleet_project = clear_fleet_project
        self.enable_security_posture = enable_security_posture
        self.network_performance_config = network_performance_config
        self.enable_k8s_beta_apis = enable_k8s_beta_apis
        self.security_posture = security_posture
        self.workload_vulnerability_scanning = workload_vulnerability_scanning
        self.enable_runtime_vulnerability_insight = enable_runtime_vulnerability_insight
        self.workload_policies = workload_policies
        self.remove_workload_policies = remove_workload_policies
        self.enable_fqdn_network_policy = enable_fqdn_network_policy
        self.host_maintenance_interval = host_maintenance_interval
        self.in_transit_encryption = in_transit_encryption
        self.enable_multi_networking = enable_multi_networking
        self.containerd_config_from_file = containerd_config_from_file
        self.autoprovisioning_resource_manager_tags = autoprovisioning_resource_manager_tags
        self.convert_to_autopilot = convert_to_autopilot
        self.convert_to_standard = convert_to_standard
        self.enable_secret_manager = enable_secret_manager
        self.enable_cilium_clusterwide_network_policy = enable_cilium_clusterwide_network_policy
        self.enable_insecure_kubelet_readonly_port = enable_insecure_kubelet_readonly_port
        self.autoprovisioning_enable_insecure_kubelet_readonly_port = autoprovisioning_enable_insecure_kubelet_readonly_port