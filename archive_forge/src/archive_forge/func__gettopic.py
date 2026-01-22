import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
def _gettopic(self, topic, more_xrefs=''):
    """Return unbuffered tuple of (topic, xrefs).

        If an error occurs here, the exception is caught and displayed by
        the url handler.

        This function duplicates the showtopic method but returns its
        result directly so it can be formatted for display in an html page.
        """
    try:
        import pydoc_data.topics
    except ImportError:
        return ('\nSorry, topic and keyword documentation is not available because the\nmodule "pydoc_data.topics" could not be found.\n', '')
    target = self.topics.get(topic, self.keywords.get(topic))
    if not target:
        raise ValueError('could not find topic')
    if isinstance(target, str):
        return self._gettopic(target, more_xrefs)
    label, xrefs = target
    doc = pydoc_data.topics.topics[label]
    if more_xrefs:
        xrefs = (xrefs or '') + ' ' + more_xrefs
    return (doc, xrefs)