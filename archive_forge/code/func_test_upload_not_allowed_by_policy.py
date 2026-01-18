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
def test_upload_not_allowed_by_policy(self):
    request = unit_test_utils.get_fake_request()
    with mock.patch.object(self.controller.policy, 'enforce') as mock_enf:
        mock_enf.side_effect = webob.exc.HTTPForbidden()
        exc = self.assertRaises(webob.exc.HTTPNotFound, self.controller.upload, request, unit_test_utils.UUID1, 'YYYY', 4)
        self.assertTrue(mock_enf.called)
    self.assertEqual('The resource could not be found.', str(exc))
    with mock.patch.object(self.controller.policy, 'enforce') as mock_enf:
        mock_enf.side_effect = [webob.exc.HTTPForbidden(), lambda *a: None]
        exc = self.assertRaises(webob.exc.HTTPForbidden, self.controller.upload, request, unit_test_utils.UUID1, 'YYYY', 4)
        self.assertTrue(mock_enf.called)