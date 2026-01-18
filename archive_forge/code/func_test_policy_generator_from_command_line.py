import os
import subprocess
from unittest import mock
import uuid
from oslo_policy import policy as common_policy
from keystone.common import policies
from keystone.common.rbac_enforcer import policy
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def test_policy_generator_from_command_line(self):
    ret_val = subprocess.Popen(['oslopolicy-policy-generator', '--namespace', 'keystone'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = ret_val.communicate()
    self.assertEqual(ret_val.returncode, 0, output)