from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
class CheckForTest(_checks.Check):

    def __call__(self, target, creds, enforcer):
        pass