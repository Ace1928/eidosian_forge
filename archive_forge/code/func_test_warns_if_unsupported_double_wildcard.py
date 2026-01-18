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
@mock.patch.object(sys.stderr, 'write', autospec=True)
def test_warns_if_unsupported_double_wildcard(self, mock_stderr):
    storage_url.StorageUrlFromString('abc**')
    storage_url.StorageUrlFromString('gs://bucket/object**')
    storage_url.StorageUrlFromString('**abc')
    storage_url.StorageUrlFromString('gs://bucket/**object')
    storage_url.StorageUrlFromString('abc**' + os.sep)
    storage_url.StorageUrlFromString('gs://bucket/object**/')
    storage_url.StorageUrlFromString(os.sep + '**abc')
    storage_url.StorageUrlFromString('gs://bucket//**object')
    storage_url.StorageUrlFromString(os.sep + '**' + os.sep + 'abc**')
    storage_url.StorageUrlFromString('gs://bucket/**/abc**')
    storage_url.StorageUrlFromString('abc**' + os.sep + 'abc')
    storage_url.StorageUrlFromString('gs://bucket/abc**/abc')
    storage_url.StorageUrlFromString(os.sep + 'abc**' + os.sep + '**')
    storage_url.StorageUrlFromString('gs://bucket/abc**/**')
    storage_url.StorageUrlFromString('gs://b**')
    storage_url.StorageUrlFromString('gs://**b')
    mock_stderr.assert_has_calls([mock.call(_UNSUPPORTED_DOUBLE_WILDCARD_WARNING_TEXT)] * 14)