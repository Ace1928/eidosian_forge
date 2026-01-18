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
def test_upload_status(self):
    request = unit_test_utils.get_fake_request(roles=['admin', 'member'])
    image = FakeImage('abcd')
    self.image_repo.result = image
    insurance = {'called': False}

    def read_data():
        insurance['called'] = True
        self.assertEqual('saving', self.image_repo.saved_image.status)
        yield 'YYYY'
    self.controller.upload(request, unit_test_utils.UUID2, read_data(), None)
    self.assertTrue(insurance['called'])
    self.assertEqual('modified-by-fake', self.image_repo.saved_image.status)