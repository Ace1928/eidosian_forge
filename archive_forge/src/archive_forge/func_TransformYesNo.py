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
def TransformYesNo(r, yes=None, no='No'):
    """Returns no if the resource is empty, yes or the resource itself otherwise.

  Args:
    r: A JSON-serializable object.
    yes: If the resource is not empty then returns _yes_ or the resource itself
      if _yes_ is not defined.
    no: Returns this value if the resource is empty.

  Returns:
    yes or r if r is not empty, no otherwise.
  """
    return (r if yes is None else yes) if r else no