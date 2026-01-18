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
def test_omit__names(self):
    ip = get_ipython()
    ip._hidden_attr = 1
    ip._x = {}
    c = ip.Completer
    ip.ex('ip=get_ipython()')
    cfg = Config()
    cfg.IPCompleter.omit__names = 0
    c.update_config(cfg)
    with provisionalcompleter():
        c.use_jedi = False
        s, matches = c.complete('ip.')
        self.assertIn('ip.__str__', matches)
        self.assertIn('ip._hidden_attr', matches)
    cfg = Config()
    cfg.IPCompleter.omit__names = 1
    c.update_config(cfg)
    with provisionalcompleter():
        c.use_jedi = False
        s, matches = c.complete('ip.')
        self.assertNotIn('ip.__str__', matches)
    cfg = Config()
    cfg.IPCompleter.omit__names = 2
    c.update_config(cfg)
    with provisionalcompleter():
        c.use_jedi = False
        s, matches = c.complete('ip.')
        self.assertNotIn('ip.__str__', matches)
        self.assertNotIn('ip._hidden_attr', matches)
    with provisionalcompleter():
        c.use_jedi = False
        s, matches = c.complete('ip._x.')
        self.assertIn('ip._x.keys', matches)
    del ip._hidden_attr
    del ip._x