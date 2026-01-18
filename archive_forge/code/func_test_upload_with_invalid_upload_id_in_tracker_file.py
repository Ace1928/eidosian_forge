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
def test_upload_with_invalid_upload_id_in_tracker_file(self):
    """
        Tests resumable upload with invalid upload ID
        """
    invalid_upload_id = 'http://pub.storage.googleapis.com/?upload_id=AyzB2Uo74W4EYxyi5dp_-r68jz8rtbvshsv4TX7srJVkJ57CxTY5Dw2'
    tmpdir = self._MakeTempDir()
    invalid_upload_id_tracker_file_name = os.path.join(tmpdir, 'invalid_upload_id_tracker')
    with open(invalid_upload_id_tracker_file_name, 'w') as f:
        f.write(invalid_upload_id)
    res_upload_handler = ResumableUploadHandler(tracker_file_name=invalid_upload_id_tracker_file_name)
    small_src_file_as_string, small_src_file = self.make_small_file()
    small_src_file.seek(0)
    dst_key = self._MakeKey(set_contents=False)
    dst_key.set_contents_from_file(small_src_file, res_upload_handler=res_upload_handler)
    self.assertEqual(SMALL_KEY_SIZE, dst_key.size)
    self.assertEqual(small_src_file_as_string, dst_key.get_contents_as_string())
    self.assertNotEqual(invalid_upload_id, res_upload_handler.get_tracker_uri())