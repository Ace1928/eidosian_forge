from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import threading
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.debug import errors
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
import six
from six.moves import urllib
def LogViewUrl(breakpoint):
    """Returns a URL to view the output for a logpoint.

  Given a breakpoint in an appengine service, this transform will return a URL
  which will open the log viewer to the request log for the service.

  Args:
    breakpoint: A breakpoint object with added information on project, service,
      debug target, and logQuery.
  Returns:
    The URL for the appropriate logs.
  """
    debug_view_url = 'https://console.cloud.google.com/logs?'
    data = [('project', breakpoint.project), ('advancedFilter', LogQueryV2String(breakpoint, separator='\n') + '\n')]
    return debug_view_url + urllib.parse.urlencode(data)