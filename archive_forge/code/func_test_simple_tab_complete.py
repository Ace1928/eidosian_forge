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
def test_simple_tab_complete(self):
    self.repl.matches_iter = MagicIterMock()
    self.repl.matches_iter.__bool__.return_value = False
    self.repl.complete = mock.Mock()
    self.repl.print_line = mock.Mock()
    self.repl.matches_iter.is_cseq.return_value = False
    self.repl.show_list = mock.Mock()
    self.repl.funcprops = mock.Mock()
    self.repl.arg_pos = mock.Mock()
    self.repl.matches_iter.cur_line.return_value = (None, 'foobar')
    self.repl.s = 'foo'
    self.repl.tab()
    self.assertTrue(self.repl.complete.called)
    self.repl.complete.assert_called_with(tab=True)
    self.assertEqual(self.repl.s, 'foobar')