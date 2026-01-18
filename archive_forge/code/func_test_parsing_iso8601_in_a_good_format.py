from argparse import ArgumentParser
from argparse import ArgumentTypeError
from unittest import mock
from datetime import datetime
from testtools import ExpectedException
from vitrageclient.tests.cli.base import CliTestCase
from vitrageclient.v1.cli.event import EventPost
def test_parsing_iso8601_in_a_good_format(self):
    self.event_post.iso8601('2014-12-13T12:44:21.123456')