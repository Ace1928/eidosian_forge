from unittest import mock
from blazarclient import base
from blazarclient import exception
from blazarclient import tests
@mock.patch('requests.request')
def test_request_fail_without_body(self, m):
    m.return_value.status_code = 400
    m.return_value.text = 'resp'
    url = '/leases'
    kwargs = {'body': {'req_key': 'req_value'}}
    self.assertRaises(exception.BlazarClientException, self.manager.request, url, 'POST', **kwargs)