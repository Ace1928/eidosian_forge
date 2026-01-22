import collections
import contextlib
import errno
import functools
import os
import sys
import types
@_instance_checking_exception(EnvironmentError)
def ChildProcessError(inst):
    return getattr(inst, 'errno', _SENTINEL) == errno.ECHILD