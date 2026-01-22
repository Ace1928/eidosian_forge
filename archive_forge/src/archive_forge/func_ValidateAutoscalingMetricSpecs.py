from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ai import constants
def ValidateAutoscalingMetricSpecs(specs):
    """Value validation for autoscaling metric specs target name and value."""
    if specs is None:
        return
    for key, value in specs.items():
        if key not in constants.OP_AUTOSCALING_METRIC_NAME_MAPPER:
            raise exceptions.InvalidArgumentException('--autoscaling-metric-specs', 'Autoscaling metric name can only be one of the following: {}.'.format(', '.join(["'{}'".format(c) for c in sorted(constants.OP_AUTOSCALING_METRIC_NAME_MAPPER.keys())])))
        if value <= 0 or value > 100:
            raise exceptions.InvalidArgumentException('--autoscaling-metric-specs', 'Metric target value %s is not between 0 and 100.' % value)