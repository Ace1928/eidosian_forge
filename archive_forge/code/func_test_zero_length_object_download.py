import errno
import os
import re
import boto
from boto.s3.resumable_download_handler import get_cur_file_size
from boto.s3.resumable_download_handler import ResumableDownloadHandler
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableDownloadException
from .cb_test_harness import CallbackTestHarness
from tests.integration.gs.testcase import GSTestCase
def test_zero_length_object_download(self):
    """
        Tests downloading a zero-length object (exercises boundary conditions).
        """
    res_download_handler = ResumableDownloadHandler()
    dst_fp = self.make_dst_fp()
    k = self._MakeKey()
    k.get_contents_to_file(dst_fp, res_download_handler=res_download_handler)
    self.assertEqual(0, get_cur_file_size(dst_fp))