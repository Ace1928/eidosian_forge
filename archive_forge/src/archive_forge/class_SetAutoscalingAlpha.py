from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute.instance_groups.managed import autoscalers as autoscalers_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class SetAutoscalingAlpha(SetAutoscaling):
    """Set autoscaling parameters of a managed instance group."""

    @staticmethod
    def Args(parser):
        managed_instance_groups_utils.AddAutoscalerArgs(parser=parser, autoscaling_file_enabled=True, patch_args=False)
        instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.AddArgument(parser)
        managed_instance_groups_utils.AddPredictiveAutoscaling(parser, standard=True)

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        managed_instance_groups_utils.ValidateAutoscalerArgs(args)
        managed_instance_groups_utils.ValidateStackdriverMetricsFlags(args)
        managed_instance_groups_utils.ValidateConflictsWithAutoscalingFile(args, managed_instance_groups_utils.ARGS_CONFLICTING_WITH_AUTOSCALING_FILE_ALPHA)
        igm_ref = instance_groups_flags.CreateGroupReference(client, holder.resources, args)
        managed_instance_groups_utils.GetInstanceGroupManagerOrThrow(igm_ref, client)
        autoscaler_resource, is_new = self.CreateAutoscalerResource(client, holder.resources, igm_ref, args)
        managed_instance_groups_utils.ValidateGeneratedAutoscalerIsValid(args, autoscaler_resource)
        autoscalers_client = autoscalers_api.GetClient(client, igm_ref)
        if args.IsSpecified('autoscaling_file'):
            if is_new:
                existing_autoscaler_name = None
            else:
                existing_autoscaler_name = autoscaler_resource.name
            return self._SetAutoscalerFromFile(args.autoscaling_file, autoscalers_client, igm_ref, existing_autoscaler_name)
        if is_new:
            managed_instance_groups_utils.AdjustAutoscalerNameForCreation(autoscaler_resource, igm_ref)
            return autoscalers_client.Insert(igm_ref, autoscaler_resource)
        return autoscalers_client.Update(igm_ref, autoscaler_resource)