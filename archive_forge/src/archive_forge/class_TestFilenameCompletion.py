import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
class TestFilenameCompletion(unittest.TestCase):

    def setUp(self):
        self.completer = autocomplete.FilenameCompletion()

    def test_locate_fails_when_not_in_string(self):
        self.assertEqual(self.completer.locate(4, 'abcd'), None)

    def test_locate_succeeds_when_in_string(self):
        self.assertEqual(self.completer.locate(4, "a'bc'd"), LinePart(2, 4, 'bc'))

    def test_issue_491(self):
        self.assertNotEqual(self.completer.matches(9, '"a[a.l-1]'), None)

    @mock.patch(glob_function, new=lambda text: [])
    def test_match_returns_none_if_not_in_string(self):
        self.assertEqual(self.completer.matches(2, 'abcd'), None)

    @mock.patch(glob_function, new=lambda text: [])
    def test_match_returns_empty_list_when_no_files(self):
        self.assertEqual(self.completer.matches(2, '"a'), set())

    @mock.patch(glob_function, new=lambda text: ['abcde', 'aaaaa'])
    @mock.patch('os.path.expanduser', new=lambda text: text)
    @mock.patch('os.path.isdir', new=lambda text: False)
    @mock.patch('os.path.sep', new='/')
    def test_match_returns_files_when_files_exist(self):
        self.assertEqual(sorted(self.completer.matches(2, '"x')), ['aaaaa', 'abcde'])

    @mock.patch(glob_function, new=lambda text: ['abcde', 'aaaaa'])
    @mock.patch('os.path.expanduser', new=lambda text: text)
    @mock.patch('os.path.isdir', new=lambda text: True)
    @mock.patch('os.path.sep', new='/')
    def test_match_returns_dirs_when_dirs_exist(self):
        self.assertEqual(sorted(self.completer.matches(2, '"x')), ['aaaaa/', 'abcde/'])

    @mock.patch(glob_function, new=lambda text: ['/expand/ed/abcde', '/expand/ed/aaaaa'])
    @mock.patch('os.path.expanduser', new=lambda text: text.replace('~', '/expand/ed'))
    @mock.patch('os.path.isdir', new=lambda text: False)
    @mock.patch('os.path.sep', new='/')
    def test_tilde_stays_pretty(self):
        self.assertEqual(sorted(self.completer.matches(4, '"~/a')), ['~/aaaaa', '~/abcde'])

    @mock.patch('os.path.sep', new='/')
    def test_formatting_takes_just_last_part(self):
        self.assertEqual(self.completer.format('/hello/there/'), 'there/')
        self.assertEqual(self.completer.format('/hello/there'), 'there')