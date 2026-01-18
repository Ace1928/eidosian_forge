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
@mock.patch('glance.common.trust_auth.TokenRefresher')
def test_upload_with_token_refresh(self, mock_refresher):
    mock_refresher.return_value = mock.MagicMock()
    mocked_save = mock.Mock()
    mocked_save.side_effect = [lambda *a: None, exception.NotAuthenticated(), lambda *a: None]
    request = unit_test_utils.get_fake_request(roles=['admin', 'member'])
    request.environ['keystone.token_info'] = {'token': {'roles': [{'name': 'member'}]}}
    image = FakeImage('abcd', owner='tenant1')
    self.image_repo.result = image
    self.image_repo.save = mocked_save
    self.controller.upload(request, unit_test_utils.UUID2, 'YYYY', 4)
    self.assertEqual('YYYY', image.data)
    self.assertEqual(4, image.size)
    self.assertEqual(3, mocked_save.call_count)