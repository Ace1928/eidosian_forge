from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.edge_cloud.container import admin_users
from googlecloudsdk.command_lib.edge_cloud.container import fleet
from googlecloudsdk.command_lib.edge_cloud.container import resource_args
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.core import resources
def PopulateClusterMessage(req, messages, args):
    """Fill the cluster message from command arguments.

  Args:
    req: create cluster request message.
    messages: message module of edgecontainer cluster.
    args: command line arguments.
  """
    req.cluster.networking = messages.ClusterNetworking()
    req.cluster.networking.clusterIpv4CidrBlocks = [args.cluster_ipv4_cidr]
    req.cluster.networking.servicesIpv4CidrBlocks = [args.services_ipv4_cidr]
    if flags.FlagIsExplicitlySet(args, 'default_max_pods_per_node'):
        req.cluster.defaultMaxPodsPerNode = int(args.default_max_pods_per_node)
    if flags.FlagIsExplicitlySet(args, 'labels'):
        req.cluster.labels = messages.Cluster.LabelsValue()
        req.cluster.labels.additionalProperties = []
        for key, value in args.labels.items():
            v = messages.Cluster.LabelsValue.AdditionalProperty()
            v.key = key
            v.value = value
            req.cluster.labels.additionalProperties.append(v)
    if flags.FlagIsExplicitlySet(args, 'maintenance_window_recurrence') or flags.FlagIsExplicitlySet(args, 'maintenance_window_start') or flags.FlagIsExplicitlySet(args, 'maintenance_window_end'):
        req.cluster.maintenancePolicy = messages.MaintenancePolicy()
        req.cluster.maintenancePolicy.window = messages.MaintenanceWindow()
        req.cluster.maintenancePolicy.window.recurringWindow = messages.RecurringTimeWindow()
        if flags.FlagIsExplicitlySet(args, 'maintenance_window_recurrence'):
            req.cluster.maintenancePolicy.window.recurringWindow.recurrence = args.maintenance_window_recurrence
        req.cluster.maintenancePolicy.window.recurringWindow.window = messages.TimeWindow()
        if flags.FlagIsExplicitlySet(args, 'maintenance_window_start'):
            req.cluster.maintenancePolicy.window.recurringWindow.window.startTime = args.maintenance_window_start
        if flags.FlagIsExplicitlySet(args, 'maintenance_window_end'):
            req.cluster.maintenancePolicy.window.recurringWindow.window.endTime = args.maintenance_window_end
    if flags.FlagIsExplicitlySet(args, 'control_plane_kms_key'):
        req.cluster.controlPlaneEncryption = messages.ControlPlaneEncryption()
        req.cluster.controlPlaneEncryption.kmsKey = args.control_plane_kms_key
    admin_users.SetAdminUsers(messages, args, req)
    fleet.SetFleetProjectPath(GetClusterReference(args), args, req)
    if flags.FlagIsExplicitlySet(args, 'external_lb_ipv4_address_pools'):
        req.cluster.externalLoadBalancerIpv4AddressPools = args.external_lb_ipv4_address_pools
    if flags.FlagIsExplicitlySet(args, 'version'):
        req.cluster.targetVersion = args.version
    if flags.FlagIsExplicitlySet(args, 'release_channel'):
        req.cluster.releaseChannel = messages.Cluster.ReleaseChannelValueValuesEnum(args.release_channel.upper())
    if flags.FlagIsExplicitlySet(args, 'control_plane_node_location') or flags.FlagIsExplicitlySet(args, 'control_plane_node_count') or flags.FlagIsExplicitlySet(args, 'control_plane_machine_filter'):
        req.cluster.controlPlane = messages.ControlPlane()
        req.cluster.controlPlane.local = messages.Local()
        if flags.FlagIsExplicitlySet(args, 'control_plane_node_location'):
            req.cluster.controlPlane.local.nodeLocation = args.control_plane_node_location
        if flags.FlagIsExplicitlySet(args, 'control_plane_node_count'):
            req.cluster.controlPlane.local.nodeCount = int(args.control_plane_node_count)
        if flags.FlagIsExplicitlySet(args, 'control_plane_machine_filter'):
            req.cluster.controlPlane.local.machineFilter = args.control_plane_machine_filter
        if flags.FlagIsExplicitlySet(args, 'control_plane_shared_deployment_policy'):
            req.cluster.controlPlane.local.sharedDeploymentPolicy = messages.Local.SharedDeploymentPolicyValueValuesEnum(args.control_plane_shared_deployment_policy.upper())