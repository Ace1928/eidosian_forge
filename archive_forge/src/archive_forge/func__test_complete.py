import os
import pytest
import sys
import textwrap
import unittest
from contextlib import contextmanager
from traitlets.config.loader import Config
from IPython import get_ipython
from IPython.core import completer
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.generics import complete_object
from IPython.testing import decorators as dec
from IPython.core.completer import (
def _test_complete(reason, s, comp, start=None, end=None):
    l = len(s)
    start = start if start is not None else l
    end = end if end is not None else l
    with provisionalcompleter():
        ip.Completer.use_jedi = True
        completions = set(ip.Completer.completions(s, l))
        ip.Completer.use_jedi = False
        assert Completion(start, end, comp) in completions, reason