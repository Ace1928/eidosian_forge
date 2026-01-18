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
def test_setmeta_seek_ahead(self):
    object_uri = self.CreateObject(contents=b'foo')
    with SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '1'), ('GSUtil', 'task_estimation_force', 'True')]):
        stderr = self.RunGsUtil(['-m', 'setmeta', '-h', 'content-type:footype', suri(object_uri)], return_stderr=True)
        self.assertIn('Estimated work for this command: objects: 1\n', stderr)