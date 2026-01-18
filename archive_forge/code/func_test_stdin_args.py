from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from gslib.cs_api_map import ApiSelector
from gslib.tests.test_cp import TestCpMvPOSIXBucketToLocalErrors
from gslib.tests.test_cp import TestCpMvPOSIXBucketToLocalNoErrors
from gslib.tests.test_cp import TestCpMvPOSIXLocalToBucketNoErrors
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SequentialAndParallelTransfer
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils import shim_util
@SequentialAndParallelTransfer
def test_stdin_args(self):
    """Tests mv with the -I option."""
    tmpdir = self.CreateTempDir()
    fpath1 = self.CreateTempFile(tmpdir=tmpdir, contents=b'data1')
    fpath2 = self.CreateTempFile(tmpdir=tmpdir, contents=b'data2')
    bucket_uri = self.CreateBucket()
    self.RunGsUtil(['mv', '-I', suri(bucket_uri)], stdin='\n'.join((fpath1, fpath2)))

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        stdout = self.RunGsUtil(['ls', suri(bucket_uri)], return_stdout=True)
        self.assertIn(os.path.basename(fpath1), stdout)
        self.assertIn(os.path.basename(fpath2), stdout)
        self.assertNumLines(stdout, 2)
    _Check1()