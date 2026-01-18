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
def test_quoted_file_completions(self):
    ip = get_ipython()

    def _(text):
        return ip.Completer._complete(cursor_line=0, cursor_pos=len(text), full_text=text)['IPCompleter.file_matcher']['completions']
    with TemporaryWorkingDirectory():
        name = "foo'bar"
        open(name, 'w', encoding='utf-8').close()
        escaped = name if sys.platform == 'win32' else "foo\\'bar"
        c = _("open('foo")[0]
        self.assertEqual(c.text, escaped)
        c = _('open("foo')[0]
        self.assertEqual(c.text, name)
        c = _('%ls foo')[0]
        self.assertEqual(c.text, escaped)