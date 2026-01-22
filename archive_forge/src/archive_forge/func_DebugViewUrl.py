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
def DebugViewUrl(breakpoint):
    """Returns a URL to view a breakpoint in the browser.

  Given a breakpoint, this transform will return a URL which will open the
  snapshot's location in a debug view pointing at the snapshot.

  Args:
    breakpoint: A breakpoint object with added information on project and
    debug target.
  Returns:
    The URL for the breakpoint.
  """
    debug_view_url = 'https://console.cloud.google.com/debug/fromgcloud?'
    data = [('project', breakpoint.project), ('dbgee', breakpoint.target_id), ('bp', breakpoint.id)]
    return debug_view_url + urllib.parse.urlencode(data)