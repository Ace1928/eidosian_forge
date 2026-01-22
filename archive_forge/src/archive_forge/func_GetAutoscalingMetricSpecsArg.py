from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetAutoscalingMetricSpecsArg():
    """Add arguments for autoscaling metric specs."""
    return base.Argument('--autoscaling-metric-specs', metavar='METRIC-NAME=TARGET', type=arg_parsers.ArgDict(key_type=str, value_type=int), action=arg_parsers.UpdateAction, help="Metric specifications that overrides a resource utilization metric's target\nvalue. At most one entry is allowed per metric.\n\n*METRIC-NAME*::: Resource metric name. Choices are {}.\n\n*TARGET*::: Target resource utilization in percentage (1% - 100%) for the\ngiven metric. If the value is set to 60, the target resource utilization is 60%.\n\nFor example:\n`--autoscaling-metric-specs=cpu-usage=70`\n".format(', '.join(["'{}'".format(c) for c in sorted(constants.OP_AUTOSCALING_METRIC_NAME_MAPPER.keys())])))