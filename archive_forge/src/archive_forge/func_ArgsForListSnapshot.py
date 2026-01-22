from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def ArgsForListSnapshot(parser):
    """Register flags for listing Cloud Dataflow snapshots.

  Args:
    parser: The argparse.ArgParser to configure with job-filtering arguments.
  """
    parser.add_argument('--job-id', required=False, metavar='JOB_ID', help='The job ID to use to filter the snapshots list.')
    parser.add_argument('--region', required=True, metavar='REGION_ID', help="The region ID of the snapshot and job's regional endpoint.")