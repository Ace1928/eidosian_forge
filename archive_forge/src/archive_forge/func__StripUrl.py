from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def _StripUrl(url):
    """Strip a http: or https: prefix, then strip leading and trailing slashes."""
    if url.startswith('https://') or url.startswith('http://'):
        return url[url.index(':') + 1:].strip('/')
    raise InvalidEndpointException(url)