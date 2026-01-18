import sys
import base64
import os.path
import unittest
import libcloud.utils.files
from libcloud.test import MockHttp, make_response, generate_random_data
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.storage.types import (
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.storage.drivers.atmos import AtmosDriver, AtmosConnection
from libcloud.storage.drivers.dummy import DummyIterator
def test_upload_object_error(self):

    def dummy_content_type(name):
        return ('application/zip', None)

    def send(self, method, **kwargs):
        raise LibcloudError('')
    old_func1 = libcloud.utils.files.guess_file_mime_type
    libcloud.utils.files.guess_file_mime_type = dummy_content_type
    old_func2 = AtmosMockHttp.request
    AtmosMockHttp.request = send
    file_path = os.path.abspath(__file__)
    container = Container(name='fbc', extra={}, driver=self)
    object_name = 'ftu'
    try:
        self.driver.upload_object(file_path=file_path, container=container, object_name=object_name)
    except LibcloudError:
        pass
    else:
        self.fail('Timeout while uploading but an exception was not thrown')
    finally:
        libcloud.utils.files.guess_file_mime_type = old_func1
        AtmosMockHttp.request = old_func2