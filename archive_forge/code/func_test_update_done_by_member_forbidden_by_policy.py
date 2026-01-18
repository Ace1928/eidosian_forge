import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_update_done_by_member_forbidden_by_policy(self):
    rules = {'modify_member': False}
    self.policy.set_rules(rules)
    request = unit_test_utils.get_fake_request(tenant=TENANT4)
    self.assertRaises(webob.exc.HTTPForbidden, self.controller.update, request, image_id=UUID2, member_id=TENANT4, status='accepted')