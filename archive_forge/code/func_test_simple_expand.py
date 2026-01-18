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
def test_simple_expand(self):
    self.repl.s = 'f'
    self.cpos = 0
    self.repl.matches_iter = mock.Mock()
    self.repl.matches_iter.is_cseq.return_value = True
    self.repl.matches_iter.substitute_cseq.return_value = (3, 'foo')
    self.repl.print_line = mock.Mock()
    self.repl.tab()
    self.assertEqual(self.repl.s, 'foo')