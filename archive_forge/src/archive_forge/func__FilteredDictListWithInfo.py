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
def _FilteredDictListWithInfo(self, result, restrict_to_type):
    """Filters a result list to contain only breakpoints of the given type.

    Args:
      result: A list of breakpoint messages, to be filtered.
      restrict_to_type: An optional breakpoint type. If None, no filtering
        will be done.
    Returns:
      The filtered result, converted to equivalent dicts with debug info fields
      added.
    """
    return [self.AddTargetInfo(r) for r in result if not restrict_to_type or r.action == self.BreakpointAction(restrict_to_type) or (not r.action and restrict_to_type == self.SNAPSHOT_TYPE)]