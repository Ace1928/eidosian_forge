from http import client as http_client
from unittest import mock
from ironicclient.common.apiclient import exceptions
from ironicclient import exc
from ironicclient.tests.unit import utils as test_utils
@mock.patch.object(exceptions, 'from_response', autospec=True)
class ExcTest(test_utils.BaseTestCase):

    def setUp(self):
        super(ExcTest, self).setUp()
        self.message = 'SpongeBob SquarePants'
        self.traceback = 'Foo Traceback'
        self.method = 'call_spongebob'
        self.url = 'http://foo.bar'
        self.expected_json = {'error': {'message': self.message, 'details': self.traceback}}

    def test_from_response(self, mock_apiclient):
        fake_response = mock.Mock(status_code=http_client.BAD_REQUEST)
        exc.from_response(fake_response, message=self.message, traceback=self.traceback, method=self.method, url=self.url)
        self.assertEqual(http_client.BAD_REQUEST, fake_response.status_code)
        self.assertEqual(self.expected_json, fake_response.json())
        mock_apiclient.assert_called_once_with(fake_response, method=self.method, url=self.url)

    def test_from_response_status(self, mock_apiclient):
        fake_response = mock.Mock(status=http_client.BAD_REQUEST)
        fake_response.getheader.return_value = 'fake-header'
        delattr(fake_response, 'status_code')
        exc.from_response(fake_response, message=self.message, traceback=self.traceback, method=self.method, url=self.url)
        expected_header = {'Content-Type': 'fake-header'}
        self.assertEqual(expected_header, fake_response.headers)
        self.assertEqual(http_client.BAD_REQUEST, fake_response.status_code)
        self.assertEqual(self.expected_json, fake_response.json())
        mock_apiclient.assert_called_once_with(fake_response, method=self.method, url=self.url)