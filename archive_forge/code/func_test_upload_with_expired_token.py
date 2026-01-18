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
def test_upload_with_expired_token(self):

    def side_effect(image, from_state=None):
        if from_state == 'saving':
            raise exception.NotAuthenticated()
    mocked_save = mock.Mock(side_effect=side_effect)
    mocked_delete = mock.Mock()
    request = unit_test_utils.get_fake_request(roles=['admin', 'member'])
    image = FakeImage('abcd')
    image.delete = mocked_delete
    self.image_repo.result = image
    self.image_repo.save = mocked_save
    self.assertRaises(webob.exc.HTTPUnauthorized, self.controller.upload, request, unit_test_utils.UUID1, 'YYYY', 4)
    self.assertEqual(3, mocked_save.call_count)
    mocked_delete.assert_called_once_with()