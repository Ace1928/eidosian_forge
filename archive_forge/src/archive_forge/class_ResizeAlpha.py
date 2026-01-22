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
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class ResizeAlpha(ResizeBeta):
    """Set managed instance group sizes."""

    @staticmethod
    def Args(parser):
        _AddArgs(parser=parser, creation_retries=True, suspended_stopped_sizes=True)
        instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.AddArgument(parser)

    @staticmethod
    def _ValidateArgs(args):
        if args.size is None and args.suspended_size is None and (args.stopped_size is None):
            raise exceptions.OneOfArgumentsRequiredException(['--size', '--suspended-size', '--stopped-size'], 'At least one of the sizes must be specified')
        if not args.creation_retries:
            if args.size is None:
                raise exceptions.RequiredArgumentException('--size', 'Size must be specified when --no-creation-retries flag is used.')
            if args.suspended_size is not None:
                raise exceptions.ConflictingArgumentsException('--suspended-size', '--no-creation-retries')
            if args.stopped_size is not None:
                raise exceptions.ConflictingArgumentsException('--stopped-size', '--no-creation-retries')

    @staticmethod
    def _MakeIgmPatchResource(client, args):
        igm_patch_resource = client.messages.InstanceGroupManager()
        if args.size is not None:
            igm_patch_resource.targetSize = args.size
        if args.suspended_size is not None:
            igm_patch_resource.targetSuspendedSize = args.suspended_size
        if args.stopped_size is not None:
            igm_patch_resource.targetStoppedSize = args.stopped_size
        return igm_patch_resource

    @staticmethod
    def _MakeIgmPatchRequest(client, igm_ref, args):
        service = client.apitools_client.instanceGroupManagers
        request = client.messages.ComputeInstanceGroupManagersPatchRequest(instanceGroupManager=igm_ref.Name(), instanceGroupManagerResource=ResizeAlpha._MakeIgmPatchResource(client, args), project=igm_ref.project, zone=igm_ref.zone)
        return client.MakeRequests([(service, 'Patch', request)])

    @staticmethod
    def _MakeRmigPatchRequest(client, igm_ref, args):
        service = client.apitools_client.regionInstanceGroupManagers
        request = client.messages.ComputeRegionInstanceGroupManagersPatchRequest(instanceGroupManager=igm_ref.Name(), instanceGroupManagerResource=ResizeAlpha._MakeIgmPatchResource(client, args), project=igm_ref.project, region=igm_ref.region)
        return client.MakeRequests([(service, 'Patch', request)])

    def Run(self, args):
        self._ValidateArgs(args)
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        igm_ref = self.CreateGroupReference(client, holder.resources, args)
        if igm_ref.Collection() == 'compute.instanceGroupManagers':
            if not args.creation_retries:
                return self._MakeIgmResizeAdvancedRequest(client, igm_ref, args)
            return self._MakeIgmPatchRequest(client, igm_ref, args)
        if igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
            if not args.creation_retries:
                return self._MakeRmigResizeAdvancedRequest(client, igm_ref, args)
            if args.suspended_size is not None or args.stopped_size is not None:
                return self._MakeRmigPatchRequest(client, igm_ref, args)
            return self._MakeRmigResizeRequest(client, igm_ref, args)
        raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))