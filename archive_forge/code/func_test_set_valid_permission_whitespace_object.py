from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
from gslib.commands import acl
from gslib.command import CreateOrGetGsutilLogger
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils import acl_helper
from gslib.utils.constants import UTF8
from gslib.utils.retry_util import Retry
from gslib.utils.translation_helper import AclTranslation
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
def test_set_valid_permission_whitespace_object(self):
    """Ensures that whitespace is allowed in role and entity elements."""
    obj_uri = suri(self.CreateObject(contents=b'foo'))
    acl_string = self.RunGsUtil(self._get_acl_prefix + [obj_uri], return_stdout=True)
    acl_string = re.sub('"role"', '"role" \\n', acl_string)
    acl_string = re.sub('"entity"', '\\n "entity"', acl_string)
    inpath = self.CreateTempFile(contents=acl_string.encode(UTF8))
    self.RunGsUtil(self._set_acl_prefix + [inpath, obj_uri])