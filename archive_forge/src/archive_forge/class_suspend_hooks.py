from __future__ import absolute_import, division, print_function
import sys
import logging
import contextlib
import copy
import os
from future.utils import PY2, PY3
class suspend_hooks(object):
    """
    Acts as a context manager. Use like this:

    >>> from future import standard_library
    >>> standard_library.install_hooks()
    >>> import http.client
    >>> # ...
    >>> with standard_library.suspend_hooks():
    >>>     import requests     # incompatible with ``future``'s standard library hooks

    If the hooks were disabled before the context, they are not installed when
    the context is left.
    """

    def __enter__(self):
        self.hooks_were_installed = detect_hooks()
        remove_hooks()
        return self

    def __exit__(self, *args):
        if self.hooks_were_installed:
            install_hooks()