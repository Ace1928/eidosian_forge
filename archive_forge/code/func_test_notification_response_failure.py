import uuid
import webob
from oslo_messaging.notify import middleware
from oslo_messaging.tests import utils
from unittest import mock
def test_notification_response_failure(self):
    m = middleware.RequestNotifier(FakeFailingApp())
    req = webob.Request.blank('/foo/bar', environ={'REQUEST_METHOD': 'GET', 'HTTP_X_AUTH_TOKEN': uuid.uuid4()})
    with mock.patch('oslo_messaging.notify.notifier.Notifier._notify') as notify:
        try:
            m(req)
            self.fail('Application exception has not been re-raised')
        except Exception:
            pass
        call_args = notify.call_args_list[0][0]
        self.assertEqual('http.request', call_args[1])
        self.assertEqual('INFO', call_args[3])
        self.assertEqual(set(['request']), set(call_args[2].keys()))
        request = call_args[2]['request']
        self.assertEqual('/foo/bar', request['PATH_INFO'])
        self.assertEqual('GET', request['REQUEST_METHOD'])
        self.assertIn('HTTP_X_SERVICE_NAME', request)
        self.assertNotIn('HTTP_X_AUTH_TOKEN', request)
        self.assertFalse(any(map(lambda s: s.startswith('wsgi.'), request.keys())), 'WSGI fields are filtered out')
        call_args = notify.call_args_list[1][0]
        self.assertEqual('http.response', call_args[1])
        self.assertEqual('INFO', call_args[3])
        self.assertEqual(set(['request', 'exception']), set(call_args[2].keys()))
        request = call_args[2]['request']
        self.assertEqual('/foo/bar', request['PATH_INFO'])
        self.assertEqual('GET', request['REQUEST_METHOD'])
        self.assertIn('HTTP_X_SERVICE_NAME', request)
        self.assertNotIn('HTTP_X_AUTH_TOKEN', request)
        self.assertFalse(any(map(lambda s: s.startswith('wsgi.'), request.keys())), 'WSGI fields are filtered out')
        exception = call_args[2]['exception']
        self.assertIn('middleware.py', exception['traceback'][0])
        self.assertIn('It happens!', exception['traceback'][-1])
        self.assertTrue(exception['value'] in ("Exception('It happens!',)", "Exception('It happens!')"))