import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
@unittest.skipUnless(has_jedi, 'jedi required')
class TestMultilineJediCompletion(unittest.TestCase):

    def test_returns_none_with_single_line(self):
        com = autocomplete.MultilineJediCompletion()
        self.assertEqual(com.matches(2, 'Va', current_block='Va', history=[]), None)

    def test_returns_non_with_blank_second_line(self):
        com = autocomplete.MultilineJediCompletion()
        self.assertEqual(com.matches(0, '', current_block='class Foo():\n', history=['class Foo():']), None)

    def matches_from_completions(self, cursor, line, block, history, completions):
        with mock.patch('bpython.autocomplete.jedi.Script') as Script:
            script = Script.return_value
            script.complete.return_value = completions
            com = autocomplete.MultilineJediCompletion()
            return com.matches(cursor, line, current_block=block, history=history)

    def test_completions_starting_with_different_letters(self):
        matches = self.matches_from_completions(2, ' a', 'class Foo:\n a', ['adsf'], [Completion('Abc', 'bc'), Completion('Cbc', 'bc')])
        self.assertEqual(matches, None)

    def test_completions_starting_with_different_cases(self):
        matches = self.matches_from_completions(2, ' a', 'class Foo:\n a', ['adsf'], [Completion('Abc', 'bc'), Completion('ade', 'de')])
        self.assertSetEqual(matches, {'ade'})

    def test_issue_544(self):
        com = autocomplete.MultilineJediCompletion()
        code = '@asyncio.coroutine\ndef'
        history = ('import asyncio', '@asyncio.coroutin')
        com.matches(3, 'def', current_block=code, history=history)