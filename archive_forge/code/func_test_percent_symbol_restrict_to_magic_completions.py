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
def test_percent_symbol_restrict_to_magic_completions(self):
    ip = get_ipython()
    completer = ip.Completer
    text = '%a'
    with provisionalcompleter():
        completer.use_jedi = True
        completions = completer.completions(text, len(text))
        for c in completions:
            self.assertEqual(c.text[0], '%')