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
def test_pub_tarball(self):
    """Ensure that the correct URI is returned based on the Python version."""
    with mock.patch.object(sys, 'version_info') as version_info:
        version_info.major = 3
        self.assertIn('gsutil.tar.gz', GsutilPubTarball())
        version_info.major = 2
        self.assertIn('gsutil4.tar.gz', GsutilPubTarball())