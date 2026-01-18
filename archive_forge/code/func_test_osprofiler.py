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
@requests_mock.Mocker()
def test_osprofiler(self, m_requests):
    self.make_env()

    def client(*args, **kwargs):
        self.assertEqual('swordfish', kwargs['profile'])
    with mock.patch('novaclient.client.Client', client):
        self.shell('list --profile swordfish', (0, 2))