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
def _WaitForOne(self, debuggee, breakpoint_id):
    try:
        breakpoint = debuggee.WaitForBreakpoint(breakpoint_id, timeout=self._timeout, completion_test=self._IsComplete)
        if not breakpoint:
            with self._result_lock:
                if not self._wait_all:
                    self._done = True
            return
        if breakpoint.isFinalState:
            with self._result_lock:
                self._results[breakpoint_id] = breakpoint
                if not self._wait_all:
                    self._done = True
    except errors.DebugError as e:
        with self._result_lock:
            self._failures.append(e)
            self._done = True