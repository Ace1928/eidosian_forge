from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.core.util import files
import requests
Return the file specified by the given request.

    Args:
      req: PreparedRequest
      **kwargs: kwargs can include values for headers, timeout, stream, etc.

    Returns:
      requests.Response object
    