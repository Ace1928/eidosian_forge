import codecs
import io
import os
import os.path
import sys
import fixtures
from oslo_config import fixture as config
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import policy
class FakeCheck(_checks.BaseCheck):

    def __init__(self, result=None):
        self.result = result

    def __str__(self):
        return str(self.result)

    def __call__(self, target, creds, enforcer):
        if self.result is not None:
            return self.result
        return (target, creds, enforcer)