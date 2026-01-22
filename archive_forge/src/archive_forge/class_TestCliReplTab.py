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
class TestCliReplTab(unittest.TestCase):

    def setUp(self):
        self.repl = FakeCliRepl()

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

    @unittest.skip('disabled while non-simple completion is disabled')
    def test_substring_tab_complete(self):
        self.repl.s = 'bar'
        self.repl.config.autocomplete_mode = autocomplete.AutocompleteModes.FUZZY
        self.repl.tab()
        self.assertEqual(self.repl.s, 'foobar')
        self.repl.tab()
        self.assertEqual(self.repl.s, 'foofoobar')

    @unittest.skip('disabled while non-simple completion is disabled')
    def test_fuzzy_tab_complete(self):
        self.repl.s = 'br'
        self.repl.config.autocomplete_mode = autocomplete.AutocompleteModes.FUZZY
        self.repl.tab()
        self.assertEqual(self.repl.s, 'foobar')

    def test_normal_tab(self):
        """make sure pressing the tab key will
        still in some cases add a tab"""
        self.repl.s = ''
        self.repl.config = mock.Mock()
        self.repl.config.tab_length = 4
        self.repl.complete = mock.Mock()
        self.repl.print_line = mock.Mock()
        self.repl.tab()
        self.assertEqual(self.repl.s, '    ')

    def test_back_parameter(self):
        self.repl.matches_iter = mock.Mock()
        self.repl.matches_iter.matches = True
        self.repl.matches_iter.previous.return_value = 'previtem'
        self.repl.matches_iter.is_cseq.return_value = False
        self.repl.show_list = mock.Mock()
        self.repl.funcprops = mock.Mock()
        self.repl.arg_pos = mock.Mock()
        self.repl.matches_iter.cur_line.return_value = (None, 'previtem')
        self.repl.print_line = mock.Mock()
        self.repl.s = 'foo'
        self.repl.cpos = 0
        self.repl.tab(back=True)
        self.assertTrue(self.repl.matches_iter.previous.called)
        self.assertTrue(self.repl.s, 'previtem')

    @unittest.skip('disabled while non-simple completion is disabled')
    def test_fuzzy_attribute_tab_complete(self):
        """Test fuzzy attribute with no text"""
        self.repl.s = 'Foo.'
        self.repl.config.autocomplete_mode = autocomplete.AutocompleteModes.FUZZY
        self.repl.tab()
        self.assertEqual(self.repl.s, 'Foo.foobar')

    @unittest.skip('disabled while non-simple completion is disabled')
    def test_fuzzy_attribute_tab_complete2(self):
        """Test fuzzy attribute with some text"""
        self.repl.s = 'Foo.br'
        self.repl.config.autocomplete_mode = autocomplete.AutocompleteModes.FUZZY
        self.repl.tab()
        self.assertEqual(self.repl.s, 'Foo.foobar')

    def test_simple_expand(self):
        self.repl.s = 'f'
        self.cpos = 0
        self.repl.matches_iter = mock.Mock()
        self.repl.matches_iter.is_cseq.return_value = True
        self.repl.matches_iter.substitute_cseq.return_value = (3, 'foo')
        self.repl.print_line = mock.Mock()
        self.repl.tab()
        self.assertEqual(self.repl.s, 'foo')

    @unittest.skip('disabled while non-simple completion is disabled')
    def test_substring_expand_forward(self):
        self.repl.config.autocomplete_mode = autocomplete.AutocompleteModes.SUBSTRING
        self.repl.s = 'ba'
        self.repl.tab()
        self.assertEqual(self.repl.s, 'bar')

    @unittest.skip('disabled while non-simple completion is disabled')
    def test_fuzzy_expand(self):
        pass