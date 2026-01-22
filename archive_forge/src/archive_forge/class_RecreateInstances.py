from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import instance_groups_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
class RecreateInstances(base.Command):
    """Recreate instances managed by a managed instance group."""

    @staticmethod
    def Args(parser):
        parser.display_info.AddFormat('\n        table(project(),\n              zone(),\n              instanceName:label=INSTANCE,\n              status)')
        _AddArgs(parser=parser)
        instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.AddArgument(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        resource_arg = instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG
        default_scope = compute_scope.ScopeEnum.ZONE
        scope_lister = flags.GetDefaultScopeLister(client)
        igm_ref = resource_arg.ResolveAsResource(args, holder.resources, default_scope=default_scope, scope_lister=scope_lister)
        if igm_ref.Collection() == 'compute.instanceGroupManagers':
            instances_holder_field = 'instanceGroupManagersRecreateInstancesRequest'
            request = client.messages.ComputeInstanceGroupManagersRecreateInstancesRequest(instanceGroupManager=igm_ref.Name(), instanceGroupManagersRecreateInstancesRequest=client.messages.InstanceGroupManagersRecreateInstancesRequest(instances=[]), project=igm_ref.project, zone=igm_ref.zone)
        elif igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
            instances_holder_field = 'regionInstanceGroupManagersRecreateRequest'
            request = client.messages.ComputeRegionInstanceGroupManagersRecreateInstancesRequest(instanceGroupManager=igm_ref.Name(), regionInstanceGroupManagersRecreateRequest=client.messages.RegionInstanceGroupManagersRecreateRequest(instances=[]), project=igm_ref.project, region=igm_ref.region)
        else:
            raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))
        return instance_groups_utils.SendInstancesRequestsAndPostProcessOutputs(api_holder=holder, method_name='RecreateInstances', request_template=request, instances_holder_field=instances_holder_field, igm_ref=igm_ref, instances=args.instances)