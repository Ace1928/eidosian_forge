from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
from gslib import storage_url
from gslib.tests import testcase
from gslib.tests import util
from gslib.tests.util import unittest
from gslib.utils import execution_util
from gslib.utils import stet_util
from unittest import mock
@mock.patch.object(stet_util, '_get_stet_binary_from_path', new=mock.Mock(return_value=None))
def test_stet_util_errors_if_no_binary(self):
    fake_config_path = self.CreateTempFile()
    source_url = storage_url.StorageUrlFromString('in')
    destination_url = storage_url.StorageUrlFromString('gs://bucket/obj')
    with util.SetBotoConfigForTest([('GSUtil', 'stet_binary_path', None), ('GSUtil', 'stet_config_path', fake_config_path)]):
        with self.assertRaises(KeyError):
            stet_util.encrypt_upload(source_url, destination_url, None)