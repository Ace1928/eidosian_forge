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
def test_upload_non_existent_image_raises_store_not_found_exception(self):

    def fake_save(self, from_state=None):
        raise glance_store.NotFound()

    def fake_delete():
        raise exception.ImageNotFound()
    request = unit_test_utils.get_fake_request(roles=['admin', 'member'])
    image = FakeImage('abcd', locations=['http://example.com/image'])
    self.image_repo.result = image
    self.image_repo.save = fake_save
    image.delete = fake_delete
    self.assertRaises(webob.exc.HTTPGone, self.controller.upload, request, str(uuid.uuid4()), 'ABC', 3)