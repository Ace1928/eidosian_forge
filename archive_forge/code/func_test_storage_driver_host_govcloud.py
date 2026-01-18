import os
import sys
import json
import tempfile
from io import BytesIO
from libcloud.test import generate_random_data  # pylint: disable-msg=E0611
from libcloud.test import unittest
from libcloud.utils.py3 import b, httplib, parse_qs, urlparse, basestring
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_AZURE_BLOBS_PARAMS, STORAGE_AZURITE_BLOBS_PARAMS
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.azure_blobs import (
def test_storage_driver_host_govcloud(self):
    driver1 = self.driver_type('fakeaccount1', 'deadbeafcafebabe==', host='blob.core.usgovcloudapi.net')
    driver2 = self.driver_type('fakeaccount2', 'deadbeafcafebabe==', host='fakeaccount2.blob.core.usgovcloudapi.net')
    host1 = driver1.connection.host
    host2 = driver2.connection.host
    account_prefix_1 = driver1.connection.account_prefix
    account_prefix_2 = driver2.connection.account_prefix
    self.assertEqual(host1, 'fakeaccount1.blob.core.usgovcloudapi.net')
    self.assertEqual(host2, 'fakeaccount2.blob.core.usgovcloudapi.net')
    self.assertIsNone(account_prefix_1)
    self.assertIsNone(account_prefix_2)