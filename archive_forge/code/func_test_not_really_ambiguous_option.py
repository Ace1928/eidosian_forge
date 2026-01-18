import argparse
import io
import re
import sys
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import fixture
import requests_mock
from testtools import matchers
from novaclient import api_versions
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import fake_actions_module
from novaclient.tests.unit import utils
def test_not_really_ambiguous_option(self):
    self.parser.add_argument('--tic-tac', action='store_true')
    self.parser.add_argument('--tic_tac', action='store_true')
    args = self.parser.parse_args(['--tic'])
    self.assertTrue(args.tic_tac)