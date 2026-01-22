from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
import uritemplate
@classmethod
def _SplitPath(cls, path):
    """Splits a GRI into its individual path segments.

    Args:
      path: str, The path segment of the GRI (from _SplitCollection)

    Returns:
      [str], A list of the path segments of the GRI.
    """
    if not path:
        return []
    parts = re.split('(?=(?<={):+[^:}]|(?<=[^:{]):+}|(?<=[^:{]):+[^:}]):', path)
    return [cls._UnescapePathSegment(part) for part in parts]