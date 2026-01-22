from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis.arg_utils import ChoiceToEnumName
def BuildAutoscaling(args, messages):
    """Build NodeGroupAutoscalingPolicy object from args."""
    autoscaling_policy = messages.NodeGroupAutoscalingPolicy(mode=messages.NodeGroupAutoscalingPolicy.ModeValueValuesEnum(ChoiceToEnumName(args.autoscaler_mode)) if args.autoscaler_mode else None, minNodes=args.min_nodes if args.IsSpecified('min_nodes') else None, maxNodes=args.max_nodes if args.IsSpecified('max_nodes') else None)
    return autoscaling_policy