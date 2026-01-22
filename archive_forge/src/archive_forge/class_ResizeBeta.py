from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class ResizeBeta(Resize):
    """Set managed instance group size."""

    @staticmethod
    def Args(parser):
        _AddArgs(parser=parser, creation_retries=True, suspended_stopped_sizes=False)
        instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.AddArgument(parser)

    @staticmethod
    def _MakeIgmResizeAdvancedRequest(client, igm_ref, args):
        service = client.apitools_client.instanceGroupManagers
        request = client.messages.ComputeInstanceGroupManagersResizeAdvancedRequest(instanceGroupManager=igm_ref.Name(), instanceGroupManagersResizeAdvancedRequest=client.messages.InstanceGroupManagersResizeAdvancedRequest(targetSize=args.size, noCreationRetries=not args.creation_retries), project=igm_ref.project, zone=igm_ref.zone)
        return client.MakeRequests([(service, 'ResizeAdvanced', request)])

    @staticmethod
    def _MakeRmigResizeAdvancedRequest(client, igm_ref, args):
        service = client.apitools_client.regionInstanceGroupManagers
        request = client.messages.ComputeRegionInstanceGroupManagersResizeAdvancedRequest(instanceGroupManager=igm_ref.Name(), regionInstanceGroupManagersResizeAdvancedRequest=client.messages.RegionInstanceGroupManagersResizeAdvancedRequest(targetSize=args.size, noCreationRetries=not args.creation_retries), project=igm_ref.project, region=igm_ref.region)
        return client.MakeRequests([(service, 'ResizeAdvanced', request)])

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        igm_ref = self.CreateGroupReference(client, holder.resources, args)
        if igm_ref.Collection() == 'compute.instanceGroupManagers':
            return self._MakeIgmResizeAdvancedRequest(client, igm_ref, args)
        if igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
            if not args.creation_retries:
                return self._MakeRmigResizeAdvancedRequest(client, igm_ref, args)
            return self._MakeRmigResizeRequest(client, igm_ref, args)
        raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))