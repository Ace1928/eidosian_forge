import os
from typing import cast
from bpython.test import FixLanguageTestCase as TestCase, TEST_CONFIG
from bpython.curtsiesfrontend import repl as curtsiesrepl
from bpython import config
from curtsies.window import CursorAwareWindow
def test_quotes(self):
    self.process_multiple_events(['(', "'", 'x', '<TAB>', ','])
    self.process_multiple_events(['[', '"', 'y', '<TAB>', '<TAB>', '<TAB>'])
    self.assertEqual(self.repl._current_line, '(\'x\',["y"])')
    self.assertEqual(self.repl._cursor_offset, 11)