from unittest import mock
import uuid
import fixtures
import webob
from keystonemiddleware.tests.unit.audit import base
def test_process_request_fail(self):
    req = webob.Request.blank('/foo/bar', environ=self.get_environ_header('GET'))
    req.environ['audit.context'] = {}
    self.create_simple_middleware()._process_request(req)
    self.assertTrue(self.notifier.notify.called)