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
def test_not_raising_error_if_multiple_buckets_with_recursion(self):
    urls = list(map(storage_url.StorageUrlFromString, ['gs://b/o', 'gs://b/p']))
    storage_url.RaiseErrorIfUrlsAreMixOfBucketsAndObjects(urls, recursion_requested=True)