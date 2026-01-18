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
@unittest.skipUnless(not gslib.IS_PACKAGE_INSTALL, 'Test is runnable only if gsutil dir is accessible, and update command is not valid for package installs.')
def test_repo_matches_manifest(self):
    """Ensure that all files/folders match the manifest."""
    tmpdir_src = self.CreateTempDir()
    gsutil_src = os.path.join(tmpdir_src, 'gsutil')
    os.makedirs(gsutil_src)
    copy_files = []
    for filename in os.listdir(GSUTIL_DIR):
        if filename.endswith('.pyc') or filename.startswith('.git') or filename == '__pycache__' or (filename == '.settings') or (filename == '.project') or (filename == '.pydevproject') or (filename == '.style.yapf') or (filename == '.yapfignore'):
            continue
        copy_files.append(filename)
    for comp in copy_files:
        if os.path.isdir(os.path.join(GSUTIL_DIR, comp)):
            func = shutil.copytree
        else:
            func = shutil.copyfile
        func(os.path.join(GSUTIL_DIR, comp), os.path.join(gsutil_src, comp))
    DisallowUpdateIfDataInGsutilDir(directory=gsutil_src)