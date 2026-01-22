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
def assertHas(self, policy, member, role):
    """Asserts a member has permission for role.

    Given an IAM policy, check if the specified member is bound to the
    specified role. Does not check group inheritence -- that is, if checking
    against the [{'member': ['allUsers'], 'role': X}] policy, this function
    will still raise an exception when testing for any member other than
    'allUsers' against role X.

    This function does not invoke the TestIamPolicy endpoints to smartly check
    IAM policy resolution. This function is simply to assert the expected IAM
    policy is returned, not whether or not the IAM policy is being invoked as
    expected.

    Args:
      policy: Policy object as formatted by IamCommand._GetIam()
      member: A member string (e.g. 'user:foo@bar.com').
      role: A fully specified role (e.g. 'roles/storage.admin')

    Raises:
      AssertionError if member is not bound to role.
    """
    policy = json.loads(policy)
    bindings = dict(((p['role'], p) for p in policy.get('bindings', [])))
    if role in bindings:
        if member in bindings[role]['members']:
            return
    raise AssertionError("Member '%s' does not have permission '%s' in policy %s" % (member, role, policy))