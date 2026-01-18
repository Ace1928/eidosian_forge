from unittest import mock
import warnings
from oslotest import base
from monascaclient import client
@mock.patch('monascaclient.client.migration')
@mock.patch('monascaclient.client._get_auth_handler')
@mock.patch('monascaclient.client._get_session')
def test_should_warn_when_passing_args(self, _, __, ___):
    api_version = mock.Mock()
    endpoint = mock.Mock()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        client.Client(api_version, endpoint)
        self.assertEqual(1, len(w))
        self.assertEqual(DeprecationWarning, w[0].category)
        self.assertRegex(str(w[0].message), 'explicit configuration of the client using')