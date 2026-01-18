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
def test_upload_object_via_stream_no_content_type(self):

    def no_content_type(name):
        return (None, None)
    old_func = libcloud.utils.files.guess_file_mime_type
    libcloud.utils.files.guess_file_mime_type = no_content_type
    container = Container(name='fbc', extra={}, driver=self)
    object_name = 'ftsdct'
    iterator = DummyIterator(data=['2', '3', '5'])
    try:
        self.driver.upload_object_via_stream(container=container, object_name=object_name, iterator=iterator)
    except AttributeError:
        pass
    else:
        self.fail('File content type not provided but an exception was not thrown')
    finally:
        libcloud.utils.files.guess_file_mime_type = old_func