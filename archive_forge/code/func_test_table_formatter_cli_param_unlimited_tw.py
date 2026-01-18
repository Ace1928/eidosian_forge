import argparse
import os
import textwrap
from io import StringIO
from unittest import mock
from cliff.formatters import table
from cliff.tests import base
from cliff.tests import test_columns
@mock.patch('cliff.utils.terminal_width')
def test_table_formatter_cli_param_unlimited_tw(self, tw):
    tw.return_value = 0
    c = ('a', 'b', 'c', 'd')
    d = ('A', 'B', 'C', 'd' * 77)
    self.assertEqual(self.expected_ml_val, _table_tester_helper(c, d, extra_args=['--max-width', '42']))