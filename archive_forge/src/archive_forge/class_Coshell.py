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
class Coshell(object):
    """The local coshell implementation shim.

  This shim class delays os specific checks until the first instantiation. The
  checks are memoized in the shim class for subsequent instantiations.
  """
    _IMPLEMENTATION = None

    def __new__(cls, *args, **kwargs):
        if not cls._IMPLEMENTATION:
            if _RunningOnWindows():
                cls._IMPLEMENTATION = _WindowsCoshell
                for shell in ['C:\\MinGW\\bin\\sh.exe', 'C:\\Program Files\\Git\\bin\\sh.exe']:
                    if os.path.isfile(shell):
                        cls._IMPLEMENTATION = _MinGWCoshell
                        cls._IMPLEMENTATION.SHELL_PATH = shell
                        break
            else:
                cls._IMPLEMENTATION = _UnixCoshell
        obj = cls._IMPLEMENTATION.__new__(cls._IMPLEMENTATION, *args, **kwargs)
        obj.__init__()
        return obj