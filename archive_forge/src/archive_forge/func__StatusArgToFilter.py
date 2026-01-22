from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import apis
from googlecloudsdk.api_lib.dataflow import job_display
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.util import times
def _StatusArgToFilter(self, status, region=None):
    """Return a string describing the job status.

    Args:
      status: The job status enum
      region: The region argument, to select the correct wrapper message.

    Returns:
      string describing the job status
    """
    filter_value_enum = None
    if region:
        filter_value_enum = apis.GetMessagesModule().DataflowProjectsLocationsJobsListRequest.FilterValueValuesEnum
    else:
        filter_value_enum = apis.GetMessagesModule().DataflowProjectsJobsAggregatedRequest.FilterValueValuesEnum
    value_map = {'all': filter_value_enum.ALL, 'terminated': filter_value_enum.TERMINATED, 'active': filter_value_enum.ACTIVE}
    return value_map.get(status, filter_value_enum.ALL)