from argparse import ArgumentParser
from argparse import ArgumentTypeError
from unittest import mock
from datetime import datetime
from testtools import ExpectedException
from vitrageclient.tests.cli.base import CliTestCase
from vitrageclient.v1.cli.event import EventPost
def test_parsing_iso8601_with_not_a_date_string(self):
    self.assertRaises(ArgumentTypeError, self.event_post.iso8601, 'bla')