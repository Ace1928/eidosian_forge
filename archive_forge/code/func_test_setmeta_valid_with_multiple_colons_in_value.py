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
def test_setmeta_valid_with_multiple_colons_in_value(self):
    obj_uri = self.CreateObject(contents=b'foo')
    self.RunGsUtil(['setmeta', '-h', 'x-%s-meta-foo:bar:baz' % self.provider_custom_meta, suri(obj_uri)])
    stdout = self.RunGsUtil(['stat', suri(obj_uri)], return_stdout=True)
    self.assertRegex(stdout, 'foo:\\s+bar:baz')