from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import platform
import re
import time
import uuid
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
def _AddQueryParam(request):
    """Sets a query parameter on a request."""
    url_parts = urllib.parse.urlsplit(request.uri)
    query_params = urllib.parse.parse_qs(url_parts.query)
    query_params[param] = value
    url_parts = list(url_parts)
    url_parts[3] = urllib.parse.urlencode(query_params, doseq=True)
    new_url = urllib.parse.urlunsplit(url_parts)
    request.uri = new_url