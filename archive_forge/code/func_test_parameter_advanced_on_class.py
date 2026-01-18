import collections
import inspect
import socket
import sys
import tempfile
import unittest
from typing import List, Tuple
from itertools import islice
from pathlib import Path
from unittest import mock
from bpython import config, repl, cli, autocomplete
from bpython.line import LinePart
from bpython.test import (
def test_parameter_advanced_on_class(self):
    self.repl = FakeRepl({'autocomplete_mode': autocomplete.AutocompleteModes.SIMPLE})
    self.set_input_line('TestCls(app')
    code = '\n        import inspect\n\n        class TestCls:\n            # A class with boring __init__ typing\n            def __init__(self, *args, **kwargs):\n                pass\n            # But that uses super exotic typings recognized by inspect.signature\n            __signature__ = inspect.Signature([\n                inspect.Parameter("apple", inspect.Parameter.POSITIONAL_ONLY),\n                inspect.Parameter("apple2", inspect.Parameter.KEYWORD_ONLY),\n                inspect.Parameter("pinetree", inspect.Parameter.KEYWORD_ONLY),\n            ])\n        '
    for line in code.split('\n'):
        print(line[8:])
        self.repl.push(line[8:])
    self.assertTrue(self.repl.complete())
    self.assertTrue(hasattr(self.repl.matches_iter, 'matches'))
    self.assertEqual(self.repl.matches_iter.matches, ['apple2=', 'apple='])