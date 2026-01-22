import argparse
import os
import textwrap
from io import StringIO
from unittest import mock
from cliff.formatters import table
from cliff.tests import base
from cliff.tests import test_columns
class TestTableFormatter(base.TestBase):

    @mock.patch('cliff.utils.terminal_width')
    def test(self, tw):
        tw.return_value = 80
        c = ('a', 'b', 'c', 'd')
        d = ('A', 'B', 'C', 'test\rcarriage\r\nreturn')
        expected = textwrap.dedent('        +-------+---------------+\n        | Field | Value         |\n        +-------+---------------+\n        | a     | A             |\n        | b     | B             |\n        | c     | C             |\n        | d     | test carriage |\n        |       | return        |\n        +-------+---------------+\n        ')
        self.assertEqual(expected, _table_tester_helper(c, d))