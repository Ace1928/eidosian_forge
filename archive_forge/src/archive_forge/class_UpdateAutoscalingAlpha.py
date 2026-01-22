from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils as mig_utils
from googlecloudsdk.api_lib.compute.instance_groups.managed import autoscalers as autoscalers_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class UpdateAutoscalingAlpha(UpdateAutoscalingBeta):
    """Update autoscaling parameters of a managed instance group."""
    clear_scale_down = True

    @staticmethod
    def Args(parser):
        _CommonArgs(parser)
        mig_utils.AddPredictiveAutoscaling(parser)
        mig_utils.AddClearScaleDownControlFlag(parser)