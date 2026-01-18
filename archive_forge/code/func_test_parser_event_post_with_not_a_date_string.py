from argparse import ArgumentParser
from argparse import ArgumentTypeError
from unittest import mock
from datetime import datetime
from testtools import ExpectedException
from vitrageclient.tests.cli.base import CliTestCase
from vitrageclient.v1.cli.event import EventPost
@mock.patch.object(ArgumentParser, 'error')
def test_parser_event_post_with_not_a_date_string(self, mock_parser):
    mock_parser.side_effect = self._my_parser_error_func
    parser = self.event_post.get_parser('vitrage event post')
    with ExpectedException(ArgumentTypeError, 'argument --time: -5 must be an iso8601 date'):
        parser.parse_args(args=['--type', 'bla', '--time', '-5', '--details', 'blabla'])