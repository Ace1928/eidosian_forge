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
def test_protect_filename():
    if sys.platform == 'win32':
        pairs = [('abc', 'abc'), (' abc', '" abc"'), ('a bc', '"a bc"'), ('a  bc', '"a  bc"'), ('  bc', '"  bc"')]
    else:
        pairs = [('abc', 'abc'), (' abc', '\\ abc'), ('a bc', 'a\\ bc'), ('a  bc', 'a\\ \\ bc'), ('  bc', '\\ \\ bc'), ('a(bc', 'a\\(bc'), ('a)bc', 'a\\)bc'), ('a( )bc', 'a\\(\\ \\)bc'), ('a[1]bc', 'a\\[1\\]bc'), ('a{1}bc', 'a\\{1\\}bc'), ('a#bc', 'a\\#bc'), ('a?bc', 'a\\?bc'), ('a=bc', 'a\\=bc'), ('a\\bc', 'a\\\\bc'), ('a|bc', 'a\\|bc'), ('a;bc', 'a\\;bc'), ('a:bc', 'a\\:bc'), ("a'bc", "a\\'bc"), ('a*bc', 'a\\*bc'), ('a"bc', 'a\\"bc'), ('a^bc', 'a\\^bc'), ('a&bc', 'a\\&bc')]
    for s1, s2 in pairs:
        s1p = completer.protect_filename(s1)
        assert s1p == s2