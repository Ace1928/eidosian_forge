from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import locale
import os
import re
import signal
import subprocess
from googlecloudsdk.core.util import encoding
import six
def SetModesCallback(self, callback):
    """Sets the callback function to be called when any mutable mode changed.

    If callback is not None then it is called immediately to initialize the
    caller.

    Args:
      callback: func() called when any mutable mode changed, None to disable.
    """
    self._set_modes_callback = callback
    if callback:
        callback()