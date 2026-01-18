import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
@mock.patch(glob_function, new=lambda text: ['/expand/ed/abcde', '/expand/ed/aaaaa'])
@mock.patch('os.path.expanduser', new=lambda text: text.replace('~', '/expand/ed'))
@mock.patch('os.path.isdir', new=lambda text: False)
@mock.patch('os.path.sep', new='/')
def test_tilde_stays_pretty(self):
    self.assertEqual(sorted(self.completer.matches(4, '"~/a')), ['~/aaaaa', '~/abcde'])