import datetime
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_trust_crud(self):
    ref = unit.new_trust_ref(trustor_user_id=self.user_id, trustee_user_id=self.trustee_user_id, project_id=self.project_id, role_ids=[self.role_id])
    r = self.post('/OS-TRUST/trusts', body={'trust': ref})
    trust = self.assertValidTrustResponse(r, ref)
    r = self.get('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']})
    self.assertValidTrustResponse(r, ref)
    r = self.get('/OS-TRUST/trusts/%(trust_id)s/roles' % {'trust_id': trust['id']})
    roles = self.assertValidRoleListResponse(r, self.role)
    self.assertIn(self.role['id'], [x['id'] for x in roles])
    self.head('/OS-TRUST/trusts/%(trust_id)s/roles/%(role_id)s' % {'trust_id': trust['id'], 'role_id': self.role['id']}, expected_status=http.client.OK)
    r = self.get('/OS-TRUST/trusts/%(trust_id)s/roles/%(role_id)s' % {'trust_id': trust['id'], 'role_id': self.role['id']})
    self.assertValidRoleResponse(r, self.role)
    r = self.get('/OS-TRUST/trusts')
    self.assertValidTrustListResponse(r, trust)
    self.delete('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']})
    self.get('/OS-TRUST/trusts/%(trust_id)s' % {'trust_id': trust['id']}, expected_status=http.client.NOT_FOUND)