import argparse
import functools
from cliff import command
from cliff.tests import base
def test_smart_help_formatter(self):
    cmd = TestCommand(None, None)
    parser = cmd.get_parser('NAME')
    parser.formatter_class = functools.partial(parser.formatter_class, width=78)
    self.assertIn(expected_help_message, parser.format_help())