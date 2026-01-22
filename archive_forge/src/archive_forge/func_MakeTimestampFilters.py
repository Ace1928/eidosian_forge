from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def MakeTimestampFilters(args):
    """Create filters for the minimum log timestamp.

  This function creates an upper bound on the timestamp of log entries.
  A filter clause is returned if order == 'desc' and timestamp is not in
  the log-filter argument.

  Args:
    args: An argparse namespace object.

  Returns:
    A list of strings that are clauses in a Cloud Logging filter expression.
  """
    if args.order == 'desc' and (not args.log_filter or 'timestamp' not in args.log_filter):
        freshness = datetime.timedelta(seconds=args.freshness)
        last_timestamp = datetime.datetime.utcnow() - freshness
        return ['timestamp>="%s"' % util.FormatTimestamp(last_timestamp)]
    else:
        return []