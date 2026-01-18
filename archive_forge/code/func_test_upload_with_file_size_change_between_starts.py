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
def test_upload_with_file_size_change_between_starts(self):
    """
        Tests resumable upload on a file that changes sizes between initial
        upload start and restart
        """
    harness = CallbackTestHarness(fail_after_n_bytes=LARGE_KEY_SIZE / 2)
    tracker_file_name = self.make_tracker_file()
    res_upload_handler = ResumableUploadHandler(tracker_file_name=tracker_file_name, num_retries=0)
    larger_src_file_as_string, larger_src_file = self.make_large_file()
    larger_src_file.seek(0)
    dst_key = self._MakeKey(set_contents=False)
    try:
        dst_key.set_contents_from_file(larger_src_file, cb=harness.call, res_upload_handler=res_upload_handler)
        self.fail('Did not get expected ResumableUploadException')
    except ResumableUploadException as e:
        self.assertEqual(e.disposition, ResumableTransferDisposition.ABORT_CUR_PROCESS)
        self.assertTrue(os.path.exists(tracker_file_name))
    time.sleep(1)
    try:
        largest_src_file = self.build_input_file(LARGEST_KEY_SIZE)[1]
        largest_src_file.seek(0)
        dst_key.set_contents_from_file(largest_src_file, res_upload_handler=res_upload_handler)
        self.fail('Did not get expected ResumableUploadException')
    except ResumableUploadException as e:
        self.assertEqual(e.disposition, ResumableTransferDisposition.ABORT)
        self.assertNotEqual(e.message.find('file size changed'), -1, e.message)