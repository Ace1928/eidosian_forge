from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
def AddMaxPercentArg(parser):
    parser.add_argument('--max-percent', help='Sets maximum percentage of instances in the group that can undergo simultaneous maintenance. If this flag is not specified default value of 1% will be set. Usage example: `--max-percent=10` sets to 10%.', type=arg_parsers.BoundedInt(lower_bound=1, upper_bound=100), default=1)