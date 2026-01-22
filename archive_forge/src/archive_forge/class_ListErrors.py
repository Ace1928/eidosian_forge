from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import instance_groups_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class ListErrors(base.ListCommand):
    """List errors produced by managed instances in a managed instance group."""

    @staticmethod
    def Args(parser):
        parser.display_info.AddFormat('        table(instanceActionDetails.instance:label=INSTANCE_URL,\n              instanceActionDetails.action:label=ACTION,\n              error.code:label=ERROR_CODE,\n              error.message:label=ERROR_MESSAGE,\n              timestamp:label=TIMESTAMP,\n              instanceActionDetails.version.instance_template:label=INSTANCE_TEMPLATE,\n              instanceActionDetails.version.name:label=VERSION_NAME\n        )')
        parser.display_info.AddUriFunc(instance_groups_utils.UriFuncForListInstanceRelatedObjects)
        instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.AddArgument(parser)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        group_ref = instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.ResolveAsResource(args, holder.resources, default_scope=compute_scope.ScopeEnum.ZONE, scope_lister=flags.GetDefaultScopeLister(client))
        if hasattr(group_ref, 'zone'):
            service = client.apitools_client.instanceGroupManagers
            request = client.messages.ComputeInstanceGroupManagersListErrorsRequest(instanceGroupManager=group_ref.Name(), zone=group_ref.zone, project=group_ref.project)
        elif hasattr(group_ref, 'region'):
            service = client.apitools_client.regionInstanceGroupManagers
            request = client.messages.ComputeRegionInstanceGroupManagersListErrorsRequest(instanceGroupManager=group_ref.Name(), region=group_ref.region, project=group_ref.project)
        batch_size = 500
        if args.page_size is not None:
            batch_size = args.page_size
        results = list_pager.YieldFromList(service, request=request, method='ListErrors', field='items', batch_size=batch_size)
        return results