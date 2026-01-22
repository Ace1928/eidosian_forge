from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves.urllib.parse import urlparse
def DeriveRegionFromEndpoint(endpoint):
    """Returns the region from a endpoint string.

  Args:
    endpoint: A str of the form `https://<region-><environment->base.url.com/`.
      Example `https://us-central-base.url.com/`,
      `https://us-central-autopush-base.url.com/`, or `https://base.url.com/`.

  Returns:
    The str region if it exists, otherwise None.
  """
    parsed = urlparse(endpoint)
    dash_splits = parsed.netloc.split('-')
    if len(dash_splits) > 2:
        return dash_splits[0] + '-' + dash_splits[1]
    else:
        return None