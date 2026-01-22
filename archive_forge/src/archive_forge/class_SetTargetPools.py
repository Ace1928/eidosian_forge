from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
class SetTargetPools(base.Command):
    """Set target pools of managed instance group.

    *{command}* sets the target pools for an existing managed instance group.
  Instances that are part of the managed instance group will be added to the
  target pool automatically.
  """

    @staticmethod
    def Args(parser):
        _AddArgs(parser=parser)
        instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.AddArgument(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        resource_arg = instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG
        default_scope = compute_scope.ScopeEnum.ZONE
        scope_lister = flags.GetDefaultScopeLister(client)
        igm_ref = resource_arg.ResolveAsResource(args, holder.resources, default_scope=default_scope, scope_lister=scope_lister)
        region = self._GetRegionName(igm_ref)
        pool_refs = []
        for target_pool in args.target_pools:
            pool_refs.append(holder.resources.Parse(target_pool, params={'project': igm_ref.project, 'region': region}, collection='compute.targetPools'))
        pools = [pool_ref.SelfLink() for pool_ref in pool_refs]
        if pools:
            return self._MakePatchRequest(client, igm_ref, pools)
        else:
            with client.apitools_client.IncludeFields(['targetPools']):
                return self._MakePatchRequest(client, igm_ref, pools)

    def _GetRegionName(self, igm_ref):
        if igm_ref.Collection() == 'compute.instanceGroupManagers':
            return utils.ZoneNameToRegionName(igm_ref.zone)
        elif igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
            return igm_ref.region
        else:
            raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))

    def _MakePatchRequest(self, client, igm_ref, pools):
        messages = client.messages
        igm_resource = messages.InstanceGroupManager(targetPools=pools)
        if igm_ref.Collection() == 'compute.instanceGroupManagers':
            service = client.apitools_client.instanceGroupManagers
            request_type = messages.ComputeInstanceGroupManagersPatchRequest
        else:
            service = client.apitools_client.regionInstanceGroupManagers
            request_type = messages.ComputeRegionInstanceGroupManagersPatchRequest
        request = request_type(**igm_ref.AsDict())
        request.instanceGroupManagerResource = igm_resource
        return client.MakeRequests([(service, 'Patch', request)])