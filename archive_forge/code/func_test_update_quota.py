import copy
from unittest import mock
from keystoneauth1 import adapter
from openstack.common import quota_set as _qs
from openstack.tests.unit import base
def test_update_quota(self):
    sot = _qs.QuotaSet.existing(project_id='proj', reservation={'a': 'b'}, usage={'c': 'd'}, foo='bar')
    resp = mock.Mock()
    resp.body = {'quota_set': copy.deepcopy(BASIC_EXAMPLE)}
    resp.json = mock.Mock(return_value=resp.body)
    resp.status_code = 200
    resp.headers = {}
    self.sess.put = mock.Mock(return_value=resp)
    sot._update(reservation={'b': 'd'}, backups=15, something_else=20)
    sot.commit(self.sess)
    self.sess.put.assert_called_with('/os-quota-sets/proj', microversion=1, headers={}, json={'quota_set': {'backups': 15, 'something_else': 20}})