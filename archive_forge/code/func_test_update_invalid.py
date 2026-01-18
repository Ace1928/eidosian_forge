import datetime
import http.client as http
import glance_store
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.image_members
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_update_invalid(self):
    request = unit_test_utils.get_fake_request()
    request.body = jsonutils.dump_as_bytes({'mem': TENANT1})
    self.assertRaises(webob.exc.HTTPBadRequest, self.deserializer.update, request)