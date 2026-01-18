import argparse
import os
import textwrap
from io import StringIO
from unittest import mock
from cliff.formatters import table
from cliff.tests import base
from cliff.tests import test_columns
@mock.patch('cliff.utils.terminal_width')
def test_table_list_formatter(self, tw):
    tw.return_value = 80
    c = ('a', 'b', 'c')
    d1 = ('A', 'B', 'C')
    d2 = ('D', 'E', 'test\rcarriage\r\nreturn')
    data = [d1, d2]
    expected = textwrap.dedent('        +---+---+---------------+\n        | a | b | c             |\n        +---+---+---------------+\n        | A | B | C             |\n        | D | E | test carriage |\n        |   |   | return        |\n        +---+---+---------------+\n        ')
    self.assertEqual(expected, _table_tester_helper(c, data))