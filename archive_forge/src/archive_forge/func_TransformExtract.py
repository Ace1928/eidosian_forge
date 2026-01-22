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
def TransformExtract(r, *keys):
    """Extract a list of non-empty values for the specified resource keys.

  Args:
    r: A JSON-serializable object.
    *keys: The list of keys in the resource whose non-empty values will be
        included in the result.

  Returns:
    The list of extracted values with empty / null values omitted.
  """
    try:
        values = [GetKeyValue(r, k, None) for k in keys]
        return [v for v in values if v]
    except TypeError:
        return []