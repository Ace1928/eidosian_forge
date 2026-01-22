from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import text
def ParseUpdateOptionsBase(args, locations):
    """Helper function to build ClusterUpdateOptions object from args.

  Args:
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.
    locations: list of strings. Zones in which cluster has nodes.

  Returns:
    ClusterUpdateOptions, object with data used to update cluster.
  """
    opts = api_adapter.UpdateClusterOptions(monitoring_service=args.monitoring_service, logging_service=args.logging_service, monitoring=args.monitoring, logging=args.logging, enable_stackdriver_kubernetes=args.enable_stackdriver_kubernetes, disable_addons=args.disable_addons, enable_autoscaling=args.enable_autoscaling, min_nodes=args.min_nodes, max_nodes=args.max_nodes, total_min_nodes=args.total_min_nodes, total_max_nodes=args.total_max_nodes, location_policy=args.location_policy, node_pool=args.node_pool, locations=locations, enable_master_authorized_networks=args.enable_master_authorized_networks, master_authorized_networks=args.master_authorized_networks, private_ipv6_google_access_type=args.private_ipv6_google_access_type, workload_pool=args.workload_pool, disable_workload_identity=args.disable_workload_identity, database_encryption_key=args.database_encryption_key, disable_database_encryption=args.disable_database_encryption, enable_vertical_pod_autoscaling=args.enable_vertical_pod_autoscaling, enable_autoprovisioning=args.enable_autoprovisioning, enable_mesh_certificates=args.enable_mesh_certificates, autoprovisioning_config_file=args.autoprovisioning_config_file, autoprovisioning_service_account=args.autoprovisioning_service_account, autoprovisioning_scopes=args.autoprovisioning_scopes, autoprovisioning_locations=args.autoprovisioning_locations, autoprovisioning_max_surge_upgrade=getattr(args, 'autoprovisioning_max_surge_upgrade', None), autoprovisioning_max_unavailable_upgrade=getattr(args, 'autoprovisioning_max_unavailable_upgrade', None), enable_autoprovisioning_surge_upgrade=getattr(args, 'enable_autoprovisioning_surge_upgrade', None), enable_autoprovisioning_blue_green_upgrade=getattr(args, 'enable_autoprovisioning_blue_green_upgrade', None), autoprovisioning_standard_rollout_policy=getattr(args, 'autoprovisioning_standard_rollout_policy', None), autoprovisioning_node_pool_soak_duration=getattr(args, 'autoprovisioning_node_pool_soak_duration', None), enable_autoprovisioning_autorepair=getattr(args, 'enable_autoprovisioning_autorepair', None), enable_autoprovisioning_autoupgrade=getattr(args, 'enable_autoprovisioning_autoupgrade', None), autoprovisioning_min_cpu_platform=getattr(args, 'autoprovisioning_min_cpu_platform', None), autoprovisioning_image_type=getattr(args, 'autoprovisioning_image_type', None), min_cpu=args.min_cpu, max_cpu=args.max_cpu, min_memory=args.min_memory, max_memory=args.max_memory, min_accelerator=args.min_accelerator, max_accelerator=args.max_accelerator, logging_variant=args.logging_variant, in_transit_encryption=getattr(args, 'in_transit_encryption', None), autoprovisioning_resource_manager_tags=args.autoprovisioning_resource_manager_tags)
    if args.disable_addons and api_adapter.GCEPDCSIDRIVER in args.disable_addons:
        pdcsi_disabled = args.disable_addons[api_adapter.GCEPDCSIDRIVER]
        if pdcsi_disabled:
            console_io.PromptContinue(message='If the GCE Persistent Disk CSI Driver is disabled, then any pods currently using PersistentVolumes owned by the driver will fail to terminate. Any new pods that try to use those PersistentVolumes will also fail to start.', cancel_on_no=True)
    if args.disable_addons and api_adapter.GCPFILESTORECSIDRIVER in args.disable_addons:
        filestorecsi_disabled = args.disable_addons[api_adapter.GCPFILESTORECSIDRIVER]
        if filestorecsi_disabled:
            console_io.PromptContinue(message='If the GCP Filestore CSI Driver is disabled, then any pods currently using PersistentVolumes owned by the driver will fail to terminate. Any new pods that try to use those PersistentVolumes will also fail to start.', cancel_on_no=True)
    if args.disable_addons and api_adapter.GCSFUSECSIDRIVER in args.disable_addons:
        gcsfusecsi_disabled = args.disable_addons[api_adapter.GCSFUSECSIDRIVER]
        if gcsfusecsi_disabled:
            console_io.PromptContinue(message='If the Cloud Storage Fuse CSI Driver is disabled, then any pods currently using PersistentVolumes owned by the driver will fail to terminate. Any new pods that try to use those PersistentVolumes will also fail to start.', cancel_on_no=True)
    if args.disable_addons and api_adapter.STATEFULHA in args.disable_addons:
        statefulha_disabled = args.disable_addons[api_adapter.STATEFULHA]
        if statefulha_disabled:
            console_io.PromptContinue(message='If the StatefulHA Addon is disabled, then any applications currently protected will no longer be updated for high availability configuration.', cancel_on_no=True)
    if args.disable_addons and api_adapter.PARALLELSTORECSIDRIVER in args.disable_addons:
        parallelstorecsi_disabled = args.disable_addons[api_adapter.PARALLELSTORECSIDRIVER]
        if parallelstorecsi_disabled:
            console_io.PromptContinue(message='If the Parallelstore CSI Driver is disabled, then any pods currently using PersistentVolumes owned by the driver will fail to terminate. Any new pods that try to use those PersistentVolumes will also fail to start.', cancel_on_no=True)
    return opts