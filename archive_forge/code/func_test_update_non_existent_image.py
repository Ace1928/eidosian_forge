import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_update_non_existent_image(self):
    request = unit_test_utils.get_fake_request(tenant=TENANT1)
    self.assertRaises(webob.exc.HTTPNotFound, self.controller.update, request, '123', TENANT4, status='accepted')