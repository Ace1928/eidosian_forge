from _pydevd_bundle.pydevd_constants import ForkSafeLock, get_global_debugger
import os
import sys
from contextlib import contextmanager
class _RedirectInfo(object):

    def __init__(self, original, redirect_to):
        self.original = original
        self.redirect_to = redirect_to