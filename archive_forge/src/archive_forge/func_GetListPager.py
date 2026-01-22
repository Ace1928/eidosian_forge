from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
import ipaddr
import six
def GetListPager(client, request, get_value_fn):
    """Returns the paged results for request from client.

  Args:
    client: The client object.
    request: The request.
    get_value_fn: Called to extract a value from an additionlProperties list
      item.

  Returns:
    The list of request results.
  """

    def _GetNextListPage():
        response = client.AggregatedList(request)
        items = []
        for item in response.items.additionalProperties:
            items += get_value_fn(item)
        return (items, response.nextPageToken)
    results, next_page_token = _GetNextListPage()
    while next_page_token:
        request.pageToken = next_page_token
        page, next_page_token = _GetNextListPage()
        results += page
    return results