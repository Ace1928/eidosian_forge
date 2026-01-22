from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib import storage_url
from gslib.tests import testcase
from gslib.utils import temporary_file_util
class TestTemporaryUtil(testcase.GsUtilUnitTestCase):
    """Test temporary file utils."""

    def testGetsTemporaryFileName(self):
        self.assertEqual(temporary_file_util.GetTempFileName(storage_url.StorageUrlFromString('file.txt')), 'file.txt_.gstmp')

    def testGetsTemporaryZipFileName(self):
        self.assertEqual(temporary_file_util.GetTempZipFileName(storage_url.StorageUrlFromString('file.txt')), 'file.txt_.gztmp')

    def testGetsStetTemporaryFileName(self):
        self.assertEqual(temporary_file_util.GetStetTempFileName(storage_url.StorageUrlFromString('file.txt')), 'file.txt_.stet_tmp')