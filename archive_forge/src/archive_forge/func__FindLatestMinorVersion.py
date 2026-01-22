from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import threading
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.debug import errors
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
import six
from six.moves import urllib
def _FindLatestMinorVersion(debuggees):
    """Given a list of debuggees, find the one with the highest minor version.

  Args:
    debuggees: A list of Debuggee objects.
  Returns:
    If all debuggees have the same name, return the one with the highest
    integer value in its 'minorversion' label. If any member of the list does
    not have a minor version, or if elements of the list have different
    names, returns None.
  """
    if not debuggees:
        return None
    best = None
    best_version = None
    name = None
    for d in debuggees:
        if not name:
            name = d.name
        elif name != d.name:
            return None
        minor_version = d.labels.get('minorversion', 0)
        if not minor_version:
            return None
        try:
            minor_version = int(minor_version)
            if not best_version or minor_version > best_version:
                best_version = minor_version
                best = d
        except ValueError:
            return None
    return best