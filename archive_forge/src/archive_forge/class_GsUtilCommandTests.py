from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import gzip
import os
import six
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import ServiceException
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.exception import NO_URLS_MATCHED_GENERIC
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetDummyProjectForUnitTest
from gslib.tests.util import unittest
from gslib.utils.constants import UTF8
from gslib.utils import copy_helper
from gslib.utils import system_util
class GsUtilCommandTests(testcase.GsUtilUnitTestCase):
    """Basic sanity check tests to make sure commands run."""

    def testDisableLoggingCommandRuns(self):
        """Test that the 'logging set off' command basically runs."""
        src_bucket_uri = self.CreateBucket()
        self.RunCommand('logging', ['set', 'off', suri(src_bucket_uri)])

    def testEnableLoggingCommandRuns(self):
        """Test that the 'logging set on' command basically runs."""
        src_bucket_uri = self.CreateBucket()
        self.RunCommand('logging', ['set', 'on', '-b', 'gs://log_bucket', suri(src_bucket_uri)])

    def testHelpCommandDoesntRaise(self):
        """Test that the help command doesn't raise (sanity checks all help)."""
        if 'PAGER' in os.environ:
            del os.environ['PAGER']
        self.RunCommand('help', [])

    def testCatCommandRuns(self):
        """Test that the cat command basically runs."""
        src_uri = self.CreateObject(contents='foo')
        stdout = self.RunCommand('cat', [suri(src_uri)], return_stdout=True)
        self.assertEqual(stdout, 'foo')

    def testGetLoggingCommandRuns(self):
        """Test that the 'logging get' command basically runs."""
        src_bucket_uri = self.CreateBucket()
        self.RunCommand('logging', ['get', suri(src_bucket_uri)])

    def testMakeBucketsCommand(self):
        """Test mb on existing bucket."""
        dst_bucket_uri = self.CreateBucket()
        try:
            with SetDummyProjectForUnitTest():
                self.RunCommand('mb', [suri(dst_bucket_uri)])
            self.fail('Did not get expected StorageCreateError')
        except ServiceException as e:
            self.assertEqual(e.status, 409)

    def testRemoveObjsCommand(self):
        """Test rm command on non-existent object."""
        dst_bucket_uri = self.CreateBucket()
        try:
            self.RunCommand('rm', [suri(dst_bucket_uri, 'non_existent')])
            self.fail('Did not get expected CommandException')
        except CommandException as e:
            self.assertIn(NO_URLS_MATCHED_TARGET % dst_bucket_uri, e.reason)