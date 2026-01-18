import copy
import testtools
from testtools import matchers
from magnumclient.tests import utils
from magnumclient.v1 import quotas
def test_quota_create(self):
    quota = self.mgr.create(**CREATE_QUOTA)
    expect = [('POST', '/v1/quotas', {}, CREATE_QUOTA)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(QUOTA1, quota._info)