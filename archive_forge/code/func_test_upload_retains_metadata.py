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
def test_upload_retains_metadata(self):
    """
        Tests that resumable upload correctly sets passed metadata
        """
    res_upload_handler = ResumableUploadHandler()
    headers = {'Content-Type': 'text/plain', 'x-goog-meta-abc': 'my meta', 'x-goog-acl': 'public-read'}
    small_src_file_as_string, small_src_file = self.make_small_file()
    small_src_file.seek(0)
    dst_key = self._MakeKey(set_contents=False)
    dst_key.set_contents_from_file(small_src_file, headers=headers, res_upload_handler=res_upload_handler)
    self.assertEqual(SMALL_KEY_SIZE, dst_key.size)
    self.assertEqual(small_src_file_as_string, dst_key.get_contents_as_string())
    dst_key.open_read()
    self.assertEqual('text/plain', dst_key.content_type)
    self.assertTrue('abc' in dst_key.metadata)
    self.assertEqual('my meta', str(dst_key.metadata['abc']))
    acl = dst_key.get_acl()
    for entry in acl.entries.entry_list:
        if str(entry.scope) == '<AllUsers>':
            self.assertEqual('READ', str(acl.entries.entry_list[1].permission))
            return
    self.fail('No <AllUsers> scope found')