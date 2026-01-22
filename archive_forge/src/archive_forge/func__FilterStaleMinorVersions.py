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
def _FilterStaleMinorVersions(debuggees):
    """Filter out any debugees referring to a stale minor version.

  Args:
    debuggees: A list of Debuggee objects.
  Returns:
    A filtered list containing only the debuggees denoting the most recent
    minor version with the given name. If any debuggee with a given name does
    not have a 'minorversion' label, the resulting list will contain all
    debuggees with that name.
  """
    byname = {}
    for debuggee in debuggees:
        if debuggee.name in byname:
            byname[debuggee.name].append(debuggee)
        else:
            byname[debuggee.name] = [debuggee]
    result = []
    for name_list in byname.values():
        latest = _FindLatestMinorVersion(name_list)
        if latest:
            result.append(latest)
        else:
            result.extend(name_list)
    return result