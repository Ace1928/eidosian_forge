from argparse import ArgumentParser
from argparse import ArgumentTypeError
from unittest import mock
import io
import json
from testtools import ExpectedException
from vitrageclient.common.formatters import DOTFormatter
from vitrageclient.common.formatters import GraphMLFormatter
from vitrageclient.tests.cli.base import CliTestCase
from vitrageclient.v1.cli.topology import TopologyShow
@mock.patch.object(ArgumentParser, 'error')
def test_parser_topology_limit_with_a_negative_number(self, mock_parser):
    mock_parser.side_effect = self._my_parser_error_func
    parser = self.topology_show.get_parser('vitrage topology show')
    with ExpectedException(ArgumentTypeError, 'argument --limit: -5 must be greater than 0'):
        parser.parse_args(args=['--filter', 'bla', '--limit', '-5', '--root', 'blabla', '--graph-type', 'tree'])