from unittest import mock
from urllib import parse
import ddt
import fixtures
from requests_mock.contrib import fixture as requests_mock_fixture
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import client
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient import utils as cinderclient_utils
from cinderclient.v3 import attachments
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
@ddt.data({'command': 'transfer-list --filters volume_id=456', 'expected': '/os-volume-transfer/detail?volume_id=456'}, {'command': 'transfer-list --filters id=123', 'expected': '/os-volume-transfer/detail?id=123'}, {'command': 'transfer-list --filters name=abc', 'expected': '/os-volume-transfer/detail?name=abc'}, {'command': 'transfer-list --filters name=abc --filters volume_id=456', 'expected': '/os-volume-transfer/detail?name=abc&volume_id=456'}, {'command': 'transfer-list --filters id=123 --filters volume_id=456', 'expected': '/os-volume-transfer/detail?id=123&volume_id=456'}, {'command': 'transfer-list --filters id=123 --filters name=abc', 'expected': '/os-volume-transfer/detail?id=123&name=abc'})
@ddt.unpack
def test_transfer_list_with_filters(self, command, expected):
    self.run_command('--os-volume-api-version 3.52 %s' % command)
    self.assert_called('GET', expected)