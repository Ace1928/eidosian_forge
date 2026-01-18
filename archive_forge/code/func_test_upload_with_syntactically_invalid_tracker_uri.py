import errno
import random
import os
import time
from six import StringIO
import boto
from boto import storage_uri
from boto.gs.resumable_upload_handler import ResumableUploadHandler
from boto.exception import InvalidUriError
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableUploadException
from .cb_test_harness import CallbackTestHarness
from tests.integration.gs.testcase import GSTestCase
def test_upload_with_syntactically_invalid_tracker_uri(self):
    """
        Tests resumable upload with a syntactically invalid tracker URI
        """
    tmp_dir = self._MakeTempDir()
    syntactically_invalid_tracker_file_name = os.path.join(tmp_dir, 'synt_invalid_uri_tracker')
    with open(syntactically_invalid_tracker_file_name, 'w') as f:
        f.write('ftp://example.com')
    res_upload_handler = ResumableUploadHandler(tracker_file_name=syntactically_invalid_tracker_file_name)
    small_src_file_as_string, small_src_file = self.make_small_file()
    small_src_file.seek(0)
    dst_key = self._MakeKey(set_contents=False)
    dst_key.set_contents_from_file(small_src_file, res_upload_handler=res_upload_handler)
    self.assertEqual(SMALL_KEY_SIZE, dst_key.size)
    self.assertEqual(small_src_file_as_string, dst_key.get_contents_as_string())