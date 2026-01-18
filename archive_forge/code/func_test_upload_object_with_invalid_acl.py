import os
import sys
import unittest
from unittest import mock
from libcloud.test import MockHttp  # pylint: disable-msg=E0611
from libcloud.test import make_response, generate_random_data
from libcloud.utils.py3 import b, httplib, parse_qs, urlparse
from libcloud.common.types import InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_OSS_PARAMS
from libcloud.storage.types import (
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.oss import CHUNK_SIZE, OSSConnection, OSSStorageDriver
from libcloud.storage.drivers.dummy import DummyIterator
def test_upload_object_with_invalid_acl(self):
    file_path = os.path.abspath(__file__)
    container = Container(name='foo_bar_container', extra={}, driver=self.driver)
    object_name = 'foo_test_upload'
    extra = {'acl': 'invalid-acl'}
    self.assertRaises(AttributeError, self.driver.upload_object, file_path=file_path, container=container, object_name=object_name, extra=extra, verify_hash=True)