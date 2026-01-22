from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests as core_requests
from googlecloudsdk.core.util import times
def MakePostRequest(url, args, extra_headers=None):
    """Makes an HTTP Post Request to the specified url with data and headers from args.

  Args:
    url: The URL to send the post request to
    args: The arguments from the command line parser
    extra_headers: Extra headers to add to the HTTP post request

  Returns:
    str: The HTTP response content
  """
    request_data = None
    headers = _DEFAULT_HEADERS
    if args.data:
        request_data = args.data
        headers = _DEFAULT_HEADERS
    elif args.cloud_event:
        request_data, headers = _StructuredToBinaryData(json.loads(args.cloud_event))
    if extra_headers:
        headers = dict(headers.items() | extra_headers.items())
    requests_session = core_requests.GetSession()
    response = requests_session.post(url=url, data=request_data, headers=headers)
    response.raise_for_status()
    return response.content