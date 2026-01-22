from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ArgsForJobRef(parser):
    """Register flags for specifying a single Job ID.

  Args:
    parser: The argparse.ArgParser to configure with job-filtering arguments.
  """
    parser.add_argument('job', metavar='JOB_ID', help='Job ID to operate on.')
    parser.add_argument('--region', metavar='REGION_ID', help="Region ID of the job's regional endpoint. " + dataflow_util.DEFAULT_REGION_MESSAGE)