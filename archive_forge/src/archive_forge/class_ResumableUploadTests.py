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
class ResumableUploadTests(GSTestCase):
    """Resumable upload test suite."""

    def build_input_file(self, size):
        buf = []
        for i in range(size):
            buf.append(str(random.randint(0, 9)))
        file_as_string = ''.join(buf)
        return (file_as_string, StringIO.StringIO(file_as_string))

    def make_small_file(self):
        return self.build_input_file(SMALL_KEY_SIZE)

    def make_large_file(self):
        return self.build_input_file(LARGE_KEY_SIZE)

    def make_tracker_file(self, tmpdir=None):
        if not tmpdir:
            tmpdir = self._MakeTempDir()
        tracker_file = os.path.join(tmpdir, 'tracker')
        return tracker_file

    def test_non_resumable_upload(self):
        """
        Tests that non-resumable uploads work
        """
        small_src_file_as_string, small_src_file = self.make_small_file()
        small_src_file.seek(0, os.SEEK_END)
        dst_key = self._MakeKey(set_contents=False)
        try:
            dst_key.set_contents_from_file(small_src_file)
            self.fail('should fail as need to rewind the filepointer')
        except AttributeError:
            pass
        dst_key.set_contents_from_file(small_src_file, rewind=True)
        self.assertEqual(SMALL_KEY_SIZE, dst_key.size)
        self.assertEqual(small_src_file_as_string, dst_key.get_contents_as_string())

    def test_upload_without_persistent_tracker(self):
        """
        Tests a single resumable upload, with no tracker URI persistence
        """
        res_upload_handler = ResumableUploadHandler()
        small_src_file_as_string, small_src_file = self.make_small_file()
        small_src_file.seek(0)
        dst_key = self._MakeKey(set_contents=False)
        dst_key.set_contents_from_file(small_src_file, res_upload_handler=res_upload_handler)
        self.assertEqual(SMALL_KEY_SIZE, dst_key.size)
        self.assertEqual(small_src_file_as_string, dst_key.get_contents_as_string())

    def test_failed_upload_with_persistent_tracker(self):
        """
        Tests that failed resumable upload leaves a correct tracker URI file
        """
        harness = CallbackTestHarness()
        tracker_file_name = self.make_tracker_file()
        res_upload_handler = ResumableUploadHandler(tracker_file_name=tracker_file_name, num_retries=0)
        small_src_file_as_string, small_src_file = self.make_small_file()
        small_src_file.seek(0)
        dst_key = self._MakeKey(set_contents=False)
        try:
            dst_key.set_contents_from_file(small_src_file, cb=harness.call, res_upload_handler=res_upload_handler)
            self.fail('Did not get expected ResumableUploadException')
        except ResumableUploadException as e:
            self.assertEqual(e.disposition, ResumableTransferDisposition.ABORT_CUR_PROCESS)
            self.assertTrue(os.path.exists(tracker_file_name))
            f = open(tracker_file_name)
            uri_from_file = f.readline().strip()
            f.close()
            self.assertEqual(uri_from_file, res_upload_handler.get_tracker_uri())

    def test_retryable_exception_recovery(self):
        """
        Tests handling of a retryable exception
        """
        exception = ResumableUploadHandler.RETRYABLE_EXCEPTIONS[0]
        harness = CallbackTestHarness(exception=exception)
        res_upload_handler = ResumableUploadHandler(num_retries=1)
        small_src_file_as_string, small_src_file = self.make_small_file()
        small_src_file.seek(0)
        dst_key = self._MakeKey(set_contents=False)
        dst_key.set_contents_from_file(small_src_file, cb=harness.call, res_upload_handler=res_upload_handler)
        self.assertEqual(SMALL_KEY_SIZE, dst_key.size)
        self.assertEqual(small_src_file_as_string, dst_key.get_contents_as_string())

    def test_broken_pipe_recovery(self):
        """
        Tests handling of a Broken Pipe (which interacts with an httplib bug)
        """
        exception = IOError(errno.EPIPE, 'Broken pipe')
        harness = CallbackTestHarness(exception=exception)
        res_upload_handler = ResumableUploadHandler(num_retries=1)
        small_src_file_as_string, small_src_file = self.make_small_file()
        small_src_file.seek(0)
        dst_key = self._MakeKey(set_contents=False)
        dst_key.set_contents_from_file(small_src_file, cb=harness.call, res_upload_handler=res_upload_handler)
        self.assertEqual(SMALL_KEY_SIZE, dst_key.size)
        self.assertEqual(small_src_file_as_string, dst_key.get_contents_as_string())

    def test_non_retryable_exception_handling(self):
        """
        Tests a resumable upload that fails with a non-retryable exception
        """
        harness = CallbackTestHarness(exception=OSError(errno.EACCES, 'Permission denied'))
        res_upload_handler = ResumableUploadHandler(num_retries=1)
        small_src_file_as_string, small_src_file = self.make_small_file()
        small_src_file.seek(0)
        dst_key = self._MakeKey(set_contents=False)
        try:
            dst_key.set_contents_from_file(small_src_file, cb=harness.call, res_upload_handler=res_upload_handler)
            self.fail('Did not get expected OSError')
        except OSError as e:
            self.assertEqual(e.errno, 13)

    def test_failed_and_restarted_upload_with_persistent_tracker(self):
        """
        Tests resumable upload that fails once and then completes, with tracker
        file
        """
        harness = CallbackTestHarness()
        tracker_file_name = self.make_tracker_file()
        res_upload_handler = ResumableUploadHandler(tracker_file_name=tracker_file_name, num_retries=1)
        small_src_file_as_string, small_src_file = self.make_small_file()
        small_src_file.seek(0)
        dst_key = self._MakeKey(set_contents=False)
        dst_key.set_contents_from_file(small_src_file, cb=harness.call, res_upload_handler=res_upload_handler)
        self.assertEqual(SMALL_KEY_SIZE, dst_key.size)
        self.assertEqual(small_src_file_as_string, dst_key.get_contents_as_string())
        self.assertFalse(os.path.exists(tracker_file_name))

    def test_multiple_in_process_failures_then_succeed(self):
        """
        Tests resumable upload that fails twice in one process, then completes
        """
        res_upload_handler = ResumableUploadHandler(num_retries=3)
        small_src_file_as_string, small_src_file = self.make_small_file()
        small_src_file.seek(0)
        dst_key = self._MakeKey(set_contents=False)
        dst_key.set_contents_from_file(small_src_file, res_upload_handler=res_upload_handler)
        self.assertEqual(SMALL_KEY_SIZE, dst_key.size)
        self.assertEqual(small_src_file_as_string, dst_key.get_contents_as_string())

    def test_multiple_in_process_failures_then_succeed_with_tracker_file(self):
        """
        Tests resumable upload that fails completely in one process,
        then when restarted completes, using a tracker file
        """
        harness = CallbackTestHarness(fail_after_n_bytes=LARGE_KEY_SIZE / 2, num_times_to_fail=2)
        tracker_file_name = self.make_tracker_file()
        res_upload_handler = ResumableUploadHandler(tracker_file_name=tracker_file_name, num_retries=1)
        larger_src_file_as_string, larger_src_file = self.make_large_file()
        larger_src_file.seek(0)
        dst_key = self._MakeKey(set_contents=False)
        try:
            dst_key.set_contents_from_file(larger_src_file, cb=harness.call, res_upload_handler=res_upload_handler)
            self.fail('Did not get expected ResumableUploadException')
        except ResumableUploadException as e:
            self.assertEqual(e.disposition, ResumableTransferDisposition.ABORT_CUR_PROCESS)
            self.assertTrue(os.path.exists(tracker_file_name))
        larger_src_file.seek(0)
        dst_key.set_contents_from_file(larger_src_file, cb=harness.call, res_upload_handler=res_upload_handler)
        self.assertEqual(LARGE_KEY_SIZE, dst_key.size)
        self.assertEqual(larger_src_file_as_string, dst_key.get_contents_as_string())
        self.assertFalse(os.path.exists(tracker_file_name))
        self.assertTrue(len(harness.transferred_seq_before_first_failure) > 1 and len(harness.transferred_seq_after_first_failure) > 1)

    def test_upload_with_inital_partial_upload_before_failure(self):
        """
        Tests resumable upload that successfully uploads some content
        before it fails, then restarts and completes
        """
        harness = CallbackTestHarness(fail_after_n_bytes=LARGE_KEY_SIZE / 2)
        res_upload_handler = ResumableUploadHandler(num_retries=1)
        larger_src_file_as_string, larger_src_file = self.make_large_file()
        larger_src_file.seek(0)
        dst_key = self._MakeKey(set_contents=False)
        dst_key.set_contents_from_file(larger_src_file, cb=harness.call, res_upload_handler=res_upload_handler)
        self.assertEqual(LARGE_KEY_SIZE, dst_key.size)
        self.assertEqual(larger_src_file_as_string, dst_key.get_contents_as_string())
        self.assertTrue(len(harness.transferred_seq_before_first_failure) > 1 and len(harness.transferred_seq_after_first_failure) > 1)

    def test_empty_file_upload(self):
        """
        Tests uploading an empty file (exercises boundary conditions).
        """
        res_upload_handler = ResumableUploadHandler()
        empty_src_file = StringIO.StringIO('')
        empty_src_file.seek(0)
        dst_key = self._MakeKey(set_contents=False)
        dst_key.set_contents_from_file(empty_src_file, res_upload_handler=res_upload_handler)
        self.assertEqual(0, dst_key.size)

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

    def test_upload_with_file_size_change_during_upload(self):
        """
        Tests resumable upload on a file that changes sizes while upload
        in progress
        """
        test_file_size = 500 * 1024
        test_file = self.build_input_file(test_file_size)[1]
        harness = CallbackTestHarness(fp_to_change=test_file, fp_change_pos=test_file_size)
        res_upload_handler = ResumableUploadHandler(num_retries=1)
        dst_key = self._MakeKey(set_contents=False)
        try:
            dst_key.set_contents_from_file(test_file, cb=harness.call, res_upload_handler=res_upload_handler)
            self.fail('Did not get expected ResumableUploadException')
        except ResumableUploadException as e:
            self.assertEqual(e.disposition, ResumableTransferDisposition.ABORT)
            self.assertNotEqual(e.message.find('File changed during upload'), -1)

    def test_upload_with_file_content_change_during_upload(self):
        """
        Tests resumable upload on a file that changes one byte of content
        (so, size stays the same) while upload in progress.
        """

        def Execute():
            res_upload_handler = ResumableUploadHandler(num_retries=1)
            dst_key = self._MakeKey(set_contents=False)
            bucket_uri = storage_uri('gs://' + dst_key.bucket.name)
            dst_key_uri = bucket_uri.clone_replace_name(dst_key.name)
            try:
                dst_key.set_contents_from_file(test_file, cb=harness.call, res_upload_handler=res_upload_handler)
                return False
            except ResumableUploadException as e:
                self.assertEqual(e.disposition, ResumableTransferDisposition.ABORT)
                test_file.seek(0, os.SEEK_END)
                self.assertEqual(test_file_size, test_file.tell())
                self.assertNotEqual(e.message.find("md5 signature doesn't match etag"), -1)
                try:
                    dst_key_uri.get_key()
                    self.fail('Did not get expected InvalidUriError')
                except InvalidUriError as e:
                    pass
            return True
        test_file_size = 500 * 1024
        n_bytes = 300 * 1024
        delay = 0
        for attempt in range(2):
            test_file = self.build_input_file(test_file_size)[1]
            harness = CallbackTestHarness(fail_after_n_bytes=n_bytes, fp_to_change=test_file, fp_change_pos=1, delay_after_change=delay)
            if Execute():
                break
            if attempt == 0 and 0 in harness.transferred_seq_after_first_failure:
                delay = 15
                continue
            self.fail('Did not get expected ResumableUploadException')

    def test_upload_with_content_length_header_set(self):
        """
        Tests resumable upload on a file when the user supplies a
        Content-Length header. This is used by gsutil, for example,
        to set the content length when gzipping a file.
        """
        res_upload_handler = ResumableUploadHandler()
        small_src_file_as_string, small_src_file = self.make_small_file()
        small_src_file.seek(0)
        dst_key = self._MakeKey(set_contents=False)
        try:
            dst_key.set_contents_from_file(small_src_file, res_upload_handler=res_upload_handler, headers={'Content-Length': SMALL_KEY_SIZE})
            self.fail('Did not get expected ResumableUploadException')
        except ResumableUploadException as e:
            self.assertEqual(e.disposition, ResumableTransferDisposition.ABORT)
            self.assertNotEqual(e.message.find('Attempt to specify Content-Length header'), -1)

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