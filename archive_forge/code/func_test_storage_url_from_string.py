from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from gslib.exception import CommandException
from gslib import storage_url
from gslib.exception import InvalidUrlError
from gslib.tests.testcase import base
from unittest import mock
def test_storage_url_from_string(self):
    url = storage_url.StorageUrlFromString('abc')
    self.assertTrue(url.IsFileUrl())
    self.assertEqual('abc', url.object_name)
    url = storage_url.StorageUrlFromString('file://abc/123')
    self.assertTrue(url.IsFileUrl())
    self.assertEqual('abc%s123' % os.sep, url.object_name)
    url = storage_url.StorageUrlFromString('gs://abc/123/456')
    self.assertTrue(url.IsCloudUrl())
    self.assertEqual('abc', url.bucket_name)
    self.assertEqual('123/456', url.object_name)
    url = storage_url.StorageUrlFromString('gs://abc///:/')
    self.assertTrue(url.IsCloudUrl())
    self.assertEqual('abc', url.bucket_name)
    self.assertEqual('//:/', url.object_name)
    url = storage_url.StorageUrlFromString('s3://abc/123/456')
    self.assertTrue(url.IsCloudUrl())
    self.assertEqual('abc', url.bucket_name)
    self.assertEqual('123/456', url.object_name)