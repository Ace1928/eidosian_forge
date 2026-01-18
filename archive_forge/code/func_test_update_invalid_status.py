import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_update_invalid_status(self):
    request = unit_test_utils.get_fake_request(tenant=TENANT4)
    self.assertRaises(webob.exc.HTTPBadRequest, self.controller.update, request, UUID2, TENANT4, status='accept')