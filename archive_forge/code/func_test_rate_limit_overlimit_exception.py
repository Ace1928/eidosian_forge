import json
from unittest import mock
import uuid
import requests
from cinderclient import client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
def test_rate_limit_overlimit_exception(self):
    cl = get_authed_client(retries=1)
    self.requests = [bad_413_request, bad_413_request, mock_request]

    @mock.patch.object(client, 'sleep', mock.Mock())
    def request(*args, **kwargs):
        next_request = self.requests.pop(0)
        return next_request(*args, **kwargs)

    @mock.patch.object(requests, 'request', request)
    @mock.patch('time.time', mock.Mock(return_value=1234))
    @mock.patch.object(client, 'sleep', mock.Mock())
    def test_get_call():
        resp, body = cl.get('/hi')
    self.assertRaises(exceptions.OverLimit, test_get_call)
    self.assertEqual([mock_request], self.requests)