from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.monitoring import completers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core.util import times
def AddUpdateableConditionFlags(parser):
    """Adds flags for condition settings that are updateable to the parser."""
    parser.add_argument('--if', dest='if_value', type=ComparisonValidator, help='One of "absent", "< THRESHOLD", "> THRESHOLD" where "THRESHOLD" is an integer or float.')
    trigger_group = parser.add_group(mutex=True)
    trigger_group.add_argument('--trigger-count', type=int, help='The absolute number of time series that must fail the predicate for the condition to be triggered.')
    trigger_group.add_argument('--trigger-percent', type=float, help='The percentage of time series that must fail the predicate for the condition to be triggered.')