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
def test_upload_storage_store_disabled(self):
    """Test that uploading an image file raises StoreDisabled exception"""
    request = unit_test_utils.get_fake_request(user=unit_test_utils.USER3, roles=['admin', 'member'])
    image = FakeImage()
    image.set_data = Raise(glance_store.StoreAddDisabled)
    self.image_repo.result = image
    self.assertRaises(webob.exc.HTTPGone, self.controller.upload, request, unit_test_utils.UUID2, 'YY', 2)