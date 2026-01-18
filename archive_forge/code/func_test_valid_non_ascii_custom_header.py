from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
from unittest import mock
import six
from gslib.commands import setmeta
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
@SkipForS3('Only ASCII characters are supported for x-amz-meta headers')
def test_valid_non_ascii_custom_header(self):
    """Tests setting custom metadata with a non-ASCII content."""
    objuri = self.CreateObject(contents=b'foo')
    unicode_header = 'x-%s-meta-dessert:soufflé' % self.provider_custom_meta
    self.RunGsUtil(['setmeta', '-h', unicode_header, suri(objuri)])

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        stdout = self.RunGsUtil(['ls', '-L', suri(objuri)], return_stdout=True)
        self.assertTrue(re.search('dessert:\\s+soufflé', stdout))
    _Check1()