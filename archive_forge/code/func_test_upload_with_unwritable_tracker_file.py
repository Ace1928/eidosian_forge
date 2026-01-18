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
def test_upload_with_unwritable_tracker_file(self):
    """
        Tests resumable upload with an unwritable tracker file
        """
    tmp_dir = self._MakeTempDir()
    tracker_file_name = self.make_tracker_file(tmp_dir)
    save_mod = os.stat(tmp_dir).st_mode
    try:
        os.chmod(tmp_dir, 0)
        res_upload_handler = ResumableUploadHandler(tracker_file_name=tracker_file_name)
    except ResumableUploadException as e:
        self.assertEqual(e.disposition, ResumableTransferDisposition.ABORT)
        self.assertNotEqual(e.message.find("Couldn't write URI tracker file"), -1)
    finally:
        os.chmod(tmp_dir, save_mod)