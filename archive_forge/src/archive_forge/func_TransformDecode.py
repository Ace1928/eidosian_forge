from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
def TransformDecode(r, encoding, undefined=''):
    """Returns the decoded value of the resource that was encoded by encoding.

  Args:
    r: A JSON-serializable object.
    encoding: The encoding name. *base64* and *utf-8* are supported.
    undefined: Returns this value if the decoding fails.

  Returns:
    The decoded resource.
  """
    if encoding == 'base64':
        try:
            return base64.urlsafe_b64decode(str(r))
        except:
            pass
    for errors in ('replace', 'strict'):
        try:
            return r.decode(encoding, errors)
        except:
            pass
    return undefined