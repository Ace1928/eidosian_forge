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
def test_upload_non_existent_image_during_save_initiates_deletion(self):

    def fake_save_not_found(self, from_state=None):
        raise exception.ImageNotFound()

    def fake_save_conflict(self, from_state=None):
        raise exception.Conflict()
    for fun in [fake_save_not_found, fake_save_conflict]:
        request = unit_test_utils.get_fake_request(roles=['admin', 'member'])
        image = FakeImage('abcd', locations=['http://example.com/image'])
        self.image_repo.result = image
        self.image_repo.save = fun
        image.delete = mock.Mock()
        self.assertRaises(webob.exc.HTTPGone, self.controller.upload, request, str(uuid.uuid4()), 'ABC', 3)
        self.assertTrue(image.delete.called)