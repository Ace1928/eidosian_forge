from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os.path
import shutil
import subprocess
import sys
import tarfile
import boto
import gslib
from gslib.metrics import _UUID_FILE_PATH
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import unittest
from gslib.utils import system_util
from gslib.utils.boto_util import CERTIFICATE_VALIDATION_ENABLED
from gslib.utils.constants import UTF8
from gslib.utils.update_util import DisallowUpdateIfDataInGsutilDir
from gslib.utils.update_util import GsutilPubTarball
from six import add_move, MovedModule
from six.moves import mock
class UpdateTest(testcase.GsUtilIntegrationTestCase):
    """Update command test suite."""

    @unittest.skipUnless(CERTIFICATE_VALIDATION_ENABLED, 'Test requires https certificate validation enabled.')
    def test_update(self):
        """Tests that the update command works or raises proper exceptions."""
        if system_util.InvokedViaCloudSdk():
            stderr = self.RunGsUtil(['update'], stdin='n', return_stderr=True, expected_status=1)
            self.assertIn('update command is disabled for Cloud SDK', stderr)
            return
        if gslib.IS_PACKAGE_INSTALL:
            stderr = self.RunGsUtil(['update'], return_stderr=True, expected_status=1)
            self.assertIn('Invalid command', stderr)
            return
        tmpdir_src = self.CreateTempDir()
        tmpdir_dst = self.CreateTempDir()
        gsutil_src = os.path.join(tmpdir_src, 'gsutil')
        gsutil_dst = os.path.join(tmpdir_dst, 'gsutil')
        gsutil_relative_dst = os.path.join('gsutil', 'gsutil')
        ignore_callable = shutil.ignore_patterns('.git*', '*.pyc', '*.pyo', '__pycache__')
        shutil.copytree(GSUTIL_DIR, gsutil_src, ignore=ignore_callable)
        os.makedirs(gsutil_dst)
        for comp in os.listdir(GSUTIL_DIR):
            if '.git' not in comp and '__pycache__' not in comp and (not comp.endswith('.pyc')) and (not comp.endswith('.pyo')):
                cp_src_path = os.path.join(GSUTIL_DIR, comp)
                cp_dst_path = os.path.join(gsutil_dst, comp)
                if os.path.isdir(cp_src_path):
                    shutil.copytree(cp_src_path, cp_dst_path, ignore=ignore_callable)
                else:
                    shutil.copyfile(cp_src_path, cp_dst_path)
        expected_version = '17.25'
        src_version_file = os.path.join(gsutil_src, 'VERSION')
        self.assertTrue(os.path.exists(src_version_file))
        with open(src_version_file, 'w') as f:
            f.write(expected_version)
        src_tarball = os.path.join(tmpdir_src, 'gsutil.test.tar.gz')
        normpath = os.path.normpath
        try:
            os.path.normpath = lambda fname: fname
            tar = tarfile.open(src_tarball, 'w:gz')
            tar.add(gsutil_src, arcname='./gsutil')
            tar.close()
        finally:
            os.path.normpath = normpath
        prefix = [sys.executable] if sys.executable else []
        p = subprocess.Popen(prefix + ['gsutil', 'update', 'gs://pub'], cwd=gsutil_dst, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = p.communicate()
        p.stdout.close()
        p.stderr.close()
        self.assertEqual(p.returncode, 1)
        self.assertIn(b'update command only works with tar.gz', stderr)
        p = subprocess.Popen(prefix + ['gsutil', 'update', 'gs://pub/Jdjh38)(;.tar.gz'], cwd=gsutil_dst, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = p.communicate()
        p.stdout.close()
        p.stderr.close()
        self.assertEqual(p.returncode, 1)
        self.assertIn(b'NotFoundException', stderr)
        p = subprocess.Popen(prefix + ['gsutil', 'update', suri(src_tarball)], cwd=gsutil_dst, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = p.communicate()
        p.stdout.close()
        p.stderr.close()
        self.assertEqual(p.returncode, 1)
        self.assertIn(b'command does not support', stderr)
        with open(os.path.join(gsutil_dst, 'userdata.txt'), 'w') as fp:
            fp.write('important data\n')
        p = subprocess.Popen(prefix + ['gsutil', 'update', '-f', suri(src_tarball)], cwd=gsutil_dst, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
        _, stderr = p.communicate()
        p.stdout.close()
        p.stderr.close()
        os.unlink(os.path.join(gsutil_dst, 'userdata.txt'))
        self.assertEqual(p.returncode, 1)
        os_ls = os.linesep.encode(UTF8)
        if os_ls in stderr:
            stderr = stderr.replace(os_ls, b' ')
        elif b'\n' in stderr:
            stderr = stderr.replace(b'\n', b' ')
        self.assertIn(b'The update command cannot run with user data in the gsutil directory', stderr)
        analytics_prompt = not (os.path.exists(_UUID_FILE_PATH) or boto.config.get_value('GSUtil', 'disable_analytics_prompt'))
        update_input = b'n\r\ny\r\n' if analytics_prompt else b'y\r\n'
        p = subprocess.Popen(prefix + [gsutil_relative_dst, 'update', '-f', suri(src_tarball)], cwd=tmpdir_dst, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
        _, stderr = p.communicate(input=update_input)
        p.stdout.close()
        p.stderr.close()
        self.assertEqual(p.returncode, 0, msg='Non-zero return code (%d) from gsutil update. stderr = \n%s' % (p.returncode, stderr.decode(UTF8)))
        dst_version_file = os.path.join(tmpdir_dst, 'gsutil', 'VERSION')
        with open(dst_version_file, 'r') as f:
            self.assertEqual(f.read(), expected_version)
        if analytics_prompt:
            os.unlink(_UUID_FILE_PATH)