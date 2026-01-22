import testtools
from unittest import mock
from glanceclient import exc
class TestHTTPExceptions(testtools.TestCase):

    def test_from_response(self):
        """exc.from_response should return instance of an HTTP exception."""
        mock_resp = mock.Mock()
        mock_resp.status_code = 400
        out = exc.from_response(mock_resp)
        self.assertIsInstance(out, exc.HTTPBadRequest)

    def test_handles_json(self):
        """exc.from_response should not print JSON."""
        mock_resp = mock.Mock()
        mock_resp.status_code = 413
        mock_resp.json.return_value = {'overLimit': {'code': 413, 'message': 'OverLimit Retry...', 'details': 'Error Details...', 'retryAt': '2014-12-03T13:33:06Z'}}
        mock_resp.headers = {'content-type': 'application/json'}
        err = exc.from_response(mock_resp, 'Non-empty body')
        self.assertIsInstance(err, exc.HTTPOverLimit)
        self.assertEqual('OverLimit Retry...', err.details)

    def test_handles_html(self):
        """exc.from_response should not print HTML."""
        mock_resp = mock.Mock()
        mock_resp.status_code = 404
        mock_resp.text = HTML_MSG
        mock_resp.headers = {'content-type': 'text/html'}
        err = exc.from_response(mock_resp, HTML_MSG)
        self.assertIsInstance(err, exc.HTTPNotFound)
        self.assertEqual('404 Entity Not Found: Entity could not be found', err.details)

    def test_format_no_content_type(self):
        mock_resp = mock.Mock()
        mock_resp.status_code = 400
        mock_resp.headers = {'content-type': 'application/octet-stream'}
        body = b'Error \n\n'
        err = exc.from_response(mock_resp, body)
        self.assertEqual('Error \n', err.details)