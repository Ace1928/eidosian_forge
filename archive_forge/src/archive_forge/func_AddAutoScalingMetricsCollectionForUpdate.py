from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddAutoScalingMetricsCollectionForUpdate(parser):
    """Adds autoscaling metrics collection flags for update.

  Args:
    parser: The argparse.parser to add the arguments to.
  """
    group = parser.add_group('Node pool autoscaling metrics collection', mutex=True)
    update_metrics_group = group.add_group('Update existing cloudwatch autoscaling metrics collection parameters')
    AddAutoscalingMetricsGranularity(update_metrics_group)
    AddAutoscalingMetrics(update_metrics_group)
    AddClearAutoscalingMetrics(group)