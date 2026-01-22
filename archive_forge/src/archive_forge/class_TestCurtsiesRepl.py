import code
import os
import sys
import tempfile
import io
from typing import cast
import unittest
from contextlib import contextmanager
from functools import partial
from unittest import mock
from bpython.curtsiesfrontend import repl as curtsiesrepl
from bpython.curtsiesfrontend import interpreter
from bpython.curtsiesfrontend import events as bpythonevents
from bpython.curtsiesfrontend.repl import LineType
from bpython import autocomplete
from bpython import config
from bpython import args
from bpython.test import (
from curtsies import events
from curtsies.window import CursorAwareWindow
from importlib import invalidate_caches
class TestCurtsiesRepl(TestCase):

    def setUp(self):
        self.repl = create_repl()

    def cfwp(self, source):
        return interpreter.code_finished_will_parse(source, self.repl.interp.compile)

    def test_code_finished_will_parse(self):
        self.repl.buffer = ['1 + 1']
        self.assertTrue(self.cfwp('\n'.join(self.repl.buffer)), (True, True))
        self.repl.buffer = ['def foo(x):']
        self.assertTrue(self.cfwp('\n'.join(self.repl.buffer)), (False, True))
        self.repl.buffer = ['def foo(x)']
        self.assertTrue(self.cfwp('\n'.join(self.repl.buffer)), (True, False))
        self.repl.buffer = ['def foo(x):', 'return 1']
        self.assertTrue(self.cfwp('\n'.join(self.repl.buffer)), (True, False))
        self.repl.buffer = ['def foo(x):', '    return 1']
        self.assertTrue(self.cfwp('\n'.join(self.repl.buffer)), (True, True))
        self.repl.buffer = ['def foo(x):', '    return 1', '']
        self.assertTrue(self.cfwp('\n'.join(self.repl.buffer)), (True, True))

    def test_external_communication(self):
        self.repl.send_current_block_to_external_editor()
        self.repl.send_session_to_external_editor()

    @unittest.skipUnless(all(map(config.can_encode, 'å∂ßƒ')), 'Charset can not encode characters')
    def test_external_communication_encoding(self):
        with captured_output():
            self.repl.display_lines.append('>>> "åß∂ƒ"')
            self.repl.history.append('"åß∂ƒ"')
            self.repl.all_logical_lines.append(('"åß∂ƒ"', LineType.INPUT))
            self.repl.send_session_to_external_editor()

    def test_get_last_word(self):
        self.repl.rl_history.entries = ['1', '2 3', '4 5 6']
        self.repl._set_current_line('abcde')
        self.repl.get_last_word()
        self.assertEqual(self.repl.current_line, 'abcde6')
        self.repl.get_last_word()
        self.assertEqual(self.repl.current_line, 'abcde3')

    def test_last_word(self):
        self.assertEqual(curtsiesrepl._last_word(''), '')
        self.assertEqual(curtsiesrepl._last_word(' '), '')
        self.assertEqual(curtsiesrepl._last_word('a'), 'a')
        self.assertEqual(curtsiesrepl._last_word('a b'), 'b')

    @unittest.skip('this is the behavior of bash - not currently implemented')
    def test_get_last_word_with_prev_line(self):
        self.repl.rl_history.entries = ['1', '2 3', '4 5 6']
        self.repl._set_current_line('abcde')
        self.repl.up_one_line()
        self.assertEqual(self.repl.current_line, '4 5 6')
        self.repl.get_last_word()
        self.assertEqual(self.repl.current_line, '4 5 63')
        self.repl.get_last_word()
        self.assertEqual(self.repl.current_line, '4 5 64')
        self.repl.up_one_line()
        self.assertEqual(self.repl.current_line, '2 3')