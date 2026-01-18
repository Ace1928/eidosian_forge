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
def test_raises_error_for_too_many_slashes_after_scheme(self):
    with self.assertRaises(InvalidUrlError):
        storage_url.StorageUrlFromString('gs:///')
    with self.assertRaises(InvalidUrlError):
        storage_url.StorageUrlFromString('gs://////')