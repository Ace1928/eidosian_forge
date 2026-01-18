from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import defaultdict
import json
import os
import subprocess
from gslib.commands import iam
from gslib.exception import CommandException
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.constants import UTF8
from gslib.utils.iam_helper import BindingsMessageToUpdateDict
from gslib.utils.iam_helper import BindingsDictToUpdateDict
from gslib.utils.iam_helper import BindingStringToTuple as bstt
from gslib.utils.iam_helper import DiffBindings
from gslib.utils.iam_helper import IsEqualBindings
from gslib.utils.iam_helper import PatchBindings
from gslib.utils.retry_util import Retry
from six import add_move, MovedModule
from six.moves import mock
def test_patch_bindings_multiple_users(self):
    """Tests expected behavior when multiple users exist."""
    expected = BindingsMessageToUpdateDict([bvle(members=['user:fii@bar.com'], role='b')])
    base = BindingsMessageToUpdateDict([bvle(members=['user:foo@bar.com'], role='a'), bvle(members=['user:foo@bar.com', 'user:fii@bar.com'], role='b'), bvle(members=['user:foo@bar.com'], role='c')])
    diff = BindingsMessageToUpdateDict([bvle(members=['user:foo@bar.com'], role='a'), bvle(members=['user:foo@bar.com'], role='b'), bvle(members=['user:foo@bar.com'], role='c')])
    res = PatchBindings(base, diff, False)
    self.assertEqual(res, expected)