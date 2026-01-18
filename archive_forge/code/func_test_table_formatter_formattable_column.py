import argparse
import os
import textwrap
from io import StringIO
from unittest import mock
from cliff.formatters import table
from cliff.tests import base
from cliff.tests import test_columns
@mock.patch('cliff.utils.terminal_width')
def test_table_formatter_formattable_column(self, tw):
    tw.return_value = 0
    c = ('a', 'b', 'c', 'd')
    d = ('A', 'B', 'C', test_columns.FauxColumn(['the', 'value']))
    expected = textwrap.dedent("        +-------+---------------------------------------------+\n        | Field | Value                                       |\n        +-------+---------------------------------------------+\n        | a     | A                                           |\n        | b     | B                                           |\n        | c     | C                                           |\n        | d     | I made this string myself: ['the', 'value'] |\n        +-------+---------------------------------------------+\n        ")
    self.assertEqual(expected, _table_tester_helper(c, d))