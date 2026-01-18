import http.client as http
import io
from unittest import mock
import uuid
from cursive import exception as cursive_exception
import glance_store
from glance_store._drivers import filesystem
from oslo_config import cfg
import webob
import glance.api.policy
import glance.api.v2.image_data
from glance.common import exception
from glance.common import wsgi
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def test_upload_invalid_backend_in_request_header(self):
    request = unit_test_utils.get_fake_request()
    request.headers['x-image-meta-store'] = 'dummy'
    image = FakeImage('abcd')
    self.image_repo.result = image
    self.assertRaises(webob.exc.HTTPBadRequest, self.controller.upload, request, unit_test_utils.UUID2, 'YYYY', 4)