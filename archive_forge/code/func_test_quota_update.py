import copy
import testtools
from testtools import matchers
from magnumclient.tests import utils
from magnumclient.v1 import quotas
def test_quota_update(self):
    patch = {'resource': 'Cluster', 'hard_limit': NEW_HARD_LIMIT, 'project_id': 'bcd'}
    quota = self.mgr.update(id=QUOTA2['project_id'], resource=QUOTA2['resource'], patch=patch)
    expect = [('PATCH', '/v1/quotas/%(id)s/%(res)s' % {'id': QUOTA2['project_id'], 'res': QUOTA2['resource']}, {}, patch)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(NEW_HARD_LIMIT, quota.hard_limit)