import argparse
import os
import textwrap
from io import StringIO
from unittest import mock
from cliff.formatters import table
from cliff.tests import base
from cliff.tests import test_columns
class TestMaxWidth(base.TestBase):
    expected_80 = textwrap.dedent('    +--------------------------+---------------------------------------------+\n    | Field                    | Value                                       |\n    +--------------------------+---------------------------------------------+\n    | field_name               | the value                                   |\n    | a_really_long_field_name | a value significantly longer than the field |\n    +--------------------------+---------------------------------------------+\n    ')

    @mock.patch('cliff.utils.terminal_width')
    def test_80(self, tw):
        tw.return_value = 80
        c = ('field_name', 'a_really_long_field_name')
        d = ('the value', 'a value significantly longer than the field')
        self.assertEqual(self.expected_80, _table_tester_helper(c, d))

    @mock.patch('cliff.utils.terminal_width')
    def test_70(self, tw):
        tw.return_value = 70
        c = ('field_name', 'a_really_long_field_name')
        d = ('the value', 'a value significantly longer than the field')
        expected = textwrap.dedent('        +--------------------------+-----------------------------------------+\n        | Field                    | Value                                   |\n        +--------------------------+-----------------------------------------+\n        | field_name               | the value                               |\n        | a_really_long_field_name | a value significantly longer than the   |\n        |                          | field                                   |\n        +--------------------------+-----------------------------------------+\n        ')
        self.assertEqual(expected, _table_tester_helper(c, d, extra_args=['--fit-width']))

    @mock.patch('cliff.utils.terminal_width')
    def test_50(self, tw):
        tw.return_value = 50
        c = ('field_name', 'a_really_long_field_name')
        d = ('the value', 'a value significantly longer than the field')
        expected = textwrap.dedent('        +-----------------------+------------------------+\n        | Field                 | Value                  |\n        +-----------------------+------------------------+\n        | field_name            | the value              |\n        | a_really_long_field_n | a value significantly  |\n        | ame                   | longer than the field  |\n        +-----------------------+------------------------+\n        ')
        self.assertEqual(expected, _table_tester_helper(c, d, extra_args=['--fit-width']))

    @mock.patch('cliff.utils.terminal_width')
    def test_10(self, tw):
        tw.return_value = 10
        c = ('field_name', 'a_really_long_field_name')
        d = ('the value', 'a value significantly longer than the field')
        expected = textwrap.dedent('        +------------------+------------------+\n        | Field            | Value            |\n        +------------------+------------------+\n        | field_name       | the value        |\n        | a_really_long_fi | a value          |\n        | eld_name         | significantly    |\n        |                  | longer than the  |\n        |                  | field            |\n        +------------------+------------------+\n        ')
        self.assertEqual(expected, _table_tester_helper(c, d, extra_args=['--fit-width']))