from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from gslib.commands import hash
from gslib.exception import CommandException
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
class TestHashUnit(testcase.GsUtilUnitTestCase):
    """Unit tests for hash command."""

    def testHashContents(self):
        tmp_file = self.CreateTempFile(contents=_TEST_FILE_CONTENTS)
        stdout = self.RunCommand('hash', args=[tmp_file], return_stdout=True)
        self.assertIn('Hashes [base64]', stdout)
        self.assertIn('\tHash (crc32c):\t\t%s' % _TEST_FILE_B64_CRC, stdout)
        self.assertIn('\tHash (md5):\t\t%s' % _TEST_FILE_B64_MD5, stdout)

    def testHashNoMatch(self):
        try:
            self.RunCommand('hash', args=['non-existent-file'])
            self.fail('Did not get expected CommandException')
        except CommandException as e:
            self.assertIn('No files matched', e.reason)

    def testHashHexFormat(self):
        tmp_file = self.CreateTempFile(contents=_TEST_FILE_CONTENTS)
        stdout = self.RunCommand('hash', args=['-h', tmp_file], return_stdout=True)
        self.assertIn('Hashes [hex]', stdout)
        self.assertIn('\tHash (crc32c):\t\t%s' % _TEST_FILE_HEX_CRC, stdout)
        self.assertIn('\tHash (md5):\t\t%s' % _TEST_FILE_HEX_MD5, stdout)

    def testHashWildcard(self):
        num_test_files = 2
        tmp_dir = self.CreateTempDir(test_files=num_test_files)
        stdout = self.RunCommand('hash', args=[os.path.join(tmp_dir, '*')], return_stdout=True)
        num_expected_lines = num_test_files * (1 + 2)
        self.assertEqual(len(stdout.splitlines()), num_expected_lines)

    def testHashSelectAlg(self):
        tmp_file = self.CreateTempFile(contents=_TEST_FILE_CONTENTS)
        stdout_crc = self.RunCommand('hash', args=['-c', tmp_file], return_stdout=True)
        stdout_md5 = self.RunCommand('hash', args=['-m', tmp_file], return_stdout=True)
        stdout_both = self.RunCommand('hash', args=['-c', '-m', tmp_file], return_stdout=True)
        for stdout in (stdout_crc, stdout_both):
            self.assertIn('\tHash (crc32c):\t\t%s' % _TEST_FILE_B64_CRC, stdout)
        for stdout in (stdout_md5, stdout_both):
            self.assertIn('\tHash (md5):\t\t%s' % _TEST_FILE_B64_MD5, stdout)
        self.assertNotIn('md5', stdout_crc)
        self.assertNotIn('crc32c', stdout_md5)