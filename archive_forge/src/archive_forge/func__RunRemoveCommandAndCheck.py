from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from unittest import mock
from gslib.exception import NO_URLS_MATCHED_PREFIX
from gslib.exception import NO_URLS_MATCHED_TARGET
import gslib.tests.testcase as testcase
from gslib.tests.testcase.base import MAX_BUCKET_LENGTH
from gslib.tests.testcase.integration_testcase import SkipForS3
import gslib.tests.util as util
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from gslib.utils.retry_util import Retry
def _RunRemoveCommandAndCheck(self, command_and_args, objects_to_remove=None, buckets_to_remove=None, stdin=None):
    """Tests a remove command in the presence of eventual listing consistency.

    Eventual listing consistency means that a remove command may not see all
    of the objects to be removed at once. When removing multiple objects
    (or buckets via -r), some calls may return no matches and multiple calls
    to the rm command may be necessary to reach the desired state. This function
    retries the rm command, incrementally tracking what has been removed and
    ensuring that the exact set of objects/buckets are removed across all
    retried calls.

    The caller is responsible for confirming the existence of buckets/objects
    prior to calling this function.

    Args:
      command_and_args: List of strings representing the rm command+args to run.
      objects_to_remove: List of object URL strings (optionally including
          generation) that should be removed by the command, if any.
      buckets_to_remove: List of bucket URL strings that should be removed by
         the command, if any.
      stdin: String of data to pipe to the process as standard input (for
         testing -I option).
    """
    bucket_strings = []
    for bucket_to_remove in buckets_to_remove or []:
        bucket_strings.append('Removing %s/...' % bucket_to_remove)
    object_strings = []
    for object_to_remove in objects_to_remove or []:
        object_strings.append('Removing %s...' % object_to_remove)
    expected_stderr_lines = set(object_strings + bucket_strings)
    if not self.multiregional_buckets and self.default_provider == 'gs':
        stderr = self.RunGsUtil(command_and_args, return_stderr=True, expected_status=None, stdin=stdin)
        num_objects = len(object_strings)
        if '-q' not in command_and_args and (not self._use_gcloud_storage):
            if '-m' in command_and_args:
                self.assertIn('[%d/%d objects]' % (num_objects, num_objects), stderr)
            else:
                self.assertIn('[%d objects]' % num_objects, stderr)
        stderr = self._CleanRmUiOutputBeforeChecking(stderr)
        stderr_set = set(stderr.splitlines())
        if '' in stderr_set:
            stderr_set.remove('')
        if MACOS_WARNING in stderr_set:
            stderr_set.remove(MACOS_WARNING)
        self.assertEqual(stderr_set, expected_stderr_lines)
    else:
        cumulative_stderr_lines = set()

        @Retry(AssertionError, tries=5, timeout_secs=1)
        def _RunRmCommandAndCheck():
            """Runs/retries the command updating+checking cumulative output."""
            stderr = self.RunGsUtil(command_and_args, return_stderr=True, expected_status=None, stdin=stdin)
            stderr = self._CleanRmUiOutputBeforeChecking(stderr)
            update_lines = True
            if NO_URLS_MATCHED_PREFIX in stderr or '409 BucketNotEmpty' in stderr or '409 VersionedBucketNotEmpty' in stderr:
                update_lines = False
            if self._use_gcloud_storage:
                bucket_not_found_string = 'not found'
            else:
                bucket_not_found_string = 'bucket does not exist'
            if '-r' in command_and_args and 'bucket does not exist' in stderr:
                for bucket_to_remove in buckets_to_remove:
                    matching_bucket = re.match('.*404\\s+%s\\s+bucket does not exist' % re.escape(bucket_to_remove), stderr)
                    if matching_bucket:
                        for line in cumulative_stderr_lines:
                            if 'Removing %s/...' % bucket_to_remove in line:
                                return
                        if 'Removing %s/...' % bucket_to_remove in stderr:
                            return
            if update_lines:
                cumulative_stderr_lines.update(set([s for s in stderr.splitlines() if s]))
            self.assertEqual(cumulative_stderr_lines, expected_stderr_lines)
        _RunRmCommandAndCheck()