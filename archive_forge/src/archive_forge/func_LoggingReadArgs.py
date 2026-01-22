from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def LoggingReadArgs(parser):
    """Arguments common to all log commands."""
    base.LIMIT_FLAG.AddToParser(parser)
    order_arg = base.ChoiceArgument('--order', choices=('desc', 'asc'), required=False, default='desc', help_str='Ordering of returned log entries based on timestamp field.')
    order_arg.AddToParser(parser)
    parser.add_argument('--freshness', type=arg_parsers.Duration(), help='Return entries that are not older than this value. Works only with DESC ordering and filters without a timestamp. See $ gcloud topic datetimes for information on duration formats.', default='1d')