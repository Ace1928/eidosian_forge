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
def _foo_bar_container_foo_test_upload_blocklist(self, method, url, body, headers):
    self._assert_content_length_header_is_string(headers=headers)
    body = ''
    headers = {}
    headers['etag'] = '0x8CFB877BB56A6FB'
    headers['content-md5'] = 'd4fe4c9829f7ca1cc89db7ad670d2bbd'
    return (httplib.CREATED, body, headers, httplib.responses[httplib.CREATED])