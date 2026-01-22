from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from random import randint
import boto
import gslib.tests.testcase as testcase
from gslib.project_id import PopulateProjectId
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.testcase.integration_testcase import SkipForJSON
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.retention_util import SECONDS_IN_DAY
from gslib.utils.retention_util import SECONDS_IN_MONTH
from gslib.utils.retention_util import SECONDS_IN_YEAR
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
def GetKey(self, mutable=False):
    keyring_fqn = self.kms_api.CreateKeyRing(PopulateProjectId(None), testcase.KmsTestingResources.KEYRING_NAME, location=testcase.KmsTestingResources.KEYRING_LOCATION)
    key_name = testcase.KmsTestingResources.CONSTANT_KEY_NAME_DO_NOT_AUTHORIZE
    if mutable:
        key_name = testcase.KmsTestingResources.MUTABLE_KEY_NAME_TEMPLATE % (randint(0, 9), randint(0, 9), randint(0, 9))
    key_fqn = self.kms_api.CreateCryptoKey(keyring_fqn, key_name)
    key_policy = self.kms_api.GetKeyIamPolicy(key_fqn)
    if key_policy.bindings:
        key_policy.bindings = []
        self.kms_api.SetKeyIamPolicy(key_fqn, key_policy)
    return key_fqn