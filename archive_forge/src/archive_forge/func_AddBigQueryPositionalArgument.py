from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from googlecloudsdk.calliope import base
def AddBigQueryPositionalArgument(parser):
    """Add BigQuery Export as a positional argument."""
    parser.add_argument('BIG_QUERY_EXPORT', metavar='BIG_QUERY_EXPORT', help='        ID of the BigQuery export e.g. `my-bq-export` or the full\n        resource name of the BigQuery export e.g.\n        `organizations/123/bigQueryExports/my-bq-export`.\n        ')
    return parser