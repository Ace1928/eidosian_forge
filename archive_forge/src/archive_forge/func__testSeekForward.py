from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import hashlib
import os
import pkgutil
from unittest import mock
from gslib.exception import CommandException
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
from gslib.utils.hashing_helper import CalculateMd5FromContents
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.hashing_helper import HashingFileUploadWrapper
def _testSeekForward(self, initial_seek):
    """Tests seeking to an initial position and then reading.

    This function simulates an upload that is resumed after a process break.
    It seeks from zero to the initial position (as if the server already had
    those bytes). Then it reads to the end of the file, ensuring the hash
    matches the original file upon completion.

    Args:
      initial_seek: Number of bytes to initially seek.

    Raises:
      AssertionError on wrong amount of data remaining or hash mismatch.
    """
    tmp_file = self._GetTestFile()
    tmp_file_len = os.path.getsize(tmp_file)
    self.assertLess(initial_seek, tmp_file_len, 'initial_seek must be less than test file size %s (but was actually: %s)' % (tmp_file_len, initial_seek))
    digesters = {'md5': GetMd5()}
    with open(tmp_file, 'rb') as stream:
        wrapper = HashingFileUploadWrapper(stream, digesters, {'md5': GetMd5}, self._dummy_url, self.logger)
        wrapper.seek(initial_seek)
        self.assertEqual(wrapper.tell(), initial_seek)
        data = wrapper.read()
        self.assertEqual(len(data), tmp_file_len - initial_seek)
    with open(tmp_file, 'rb') as stream:
        actual = CalculateMd5FromContents(stream)
    self.assertEqual(actual, digesters['md5'].hexdigest())