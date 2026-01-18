import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_update_done_by_member_allowed_by_policy(self):
    rules = {'modify_member': True}
    self.policy.set_rules(rules)
    request = unit_test_utils.get_fake_request(tenant=TENANT4)
    output = self.controller.update(request, image_id=UUID2, member_id=TENANT4, status='accepted')
    self.assertEqual(UUID2, output.image_id)
    self.assertEqual(TENANT4, output.member_id)
    self.assertEqual('accepted', output.status)