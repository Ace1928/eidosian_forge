from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def GetParamsFromPath(path):
    """Extract parameters from uri template path.

    See https://tools.ietf.org/html/rfc6570. This function makes simplifying
    assumption that all parameter names are surrounded by /{ and }/.

  Args:
    path: str, uri template path.

  Returns:
    list(str), list of parameters in the template path.
  """
    path = path.split(':')[0]
    parts = path.split('/')
    params = []
    for part in parts:
        if part.startswith('{') and part.endswith('}'):
            part = part[1:-1]
            if part.startswith('+'):
                params.append(part[1:])
            else:
                params.append(part)
    return params