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
def test_urls_raise_error_if_object_followed_by_bucket(self):
    urls = list(map(storage_url.StorageUrlFromString, ['gs://b/o', 'gs://b']))
    with self.assertRaisesRegex(CommandException, 'Cannot operate on a mix of buckets and objects.'):
        storage_url.RaiseErrorIfUrlsAreMixOfBucketsAndObjects(urls, recursion_requested=False)