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
def _test_upload_download_prepare_notification(self):
    request = unit_test_utils.get_fake_request()
    self.controller.upload(request, unit_test_utils.UUID2, 'YYYY', 4)
    output = self.controller.download(request, unit_test_utils.UUID2)
    output_log = self.notifier.get_logs()
    prepare_payload = output['meta'].copy()
    prepare_payload['checksum'] = None
    prepare_payload['size'] = None
    prepare_payload['virtual_size'] = None
    prepare_payload['location'] = None
    prepare_payload['status'] = 'queued'
    del prepare_payload['updated_at']
    prepare_log = {'notification_type': 'INFO', 'event_type': 'image.prepare', 'payload': prepare_payload}
    self.assertEqual(3, len(output_log))
    prepare_updated_at = output_log[0]['payload']['updated_at']
    del output_log[0]['payload']['updated_at']
    self.assertLessEqual(prepare_updated_at, output['meta']['updated_at'])
    self.assertEqual(prepare_log, output_log[0])