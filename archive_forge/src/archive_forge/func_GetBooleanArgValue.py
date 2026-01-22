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
def GetBooleanArgValue(arg):
    """Returns the Boolean value for arg."""
    if arg in (True, False):
        return arg
    if not arg:
        return False
    try:
        if arg.lower() == 'false':
            return False
    except AttributeError:
        pass
    try:
        return bool(float(arg))
    except ValueError:
        pass
    return True