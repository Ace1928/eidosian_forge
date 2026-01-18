import copy
import testtools
from testtools import matchers
from magnumclient.tests import utils
from magnumclient.v1 import quotas
def test_quota_delete(self):
    quota = self.mgr.delete(QUOTA2['project_id'], QUOTA2['resource'])
    expect = [('DELETE', '/v1/quotas/%(id)s/%(res)s' % {'id': QUOTA2['project_id'], 'res': QUOTA2['resource']}, {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(quota)