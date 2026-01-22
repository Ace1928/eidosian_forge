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
def MaybePrependToHeader(header, value):
    """Prepends the given value if the existing header does not start with it.

  Args:
    header: str, The name of the header to prepend to.
    value: str, The value to prepend to the existing header value.

  Returns:
    A function that can be used in a Handler.request.
  """
    header, value = _EncodeHeader(header, value)

    def _MaybePrependToHeader(request):
        """Maybe prepends a value to a header on a request."""
        headers = request.headers
        current_value = b''
        for hdr, v in six.iteritems(headers):
            if hdr.lower() == header.lower():
                current_value = v
                del headers[hdr]
                break
        if not current_value.startswith(value):
            current_value = (value + b' ' + current_value).strip()
        headers[header] = current_value
    return _MaybePrependToHeader