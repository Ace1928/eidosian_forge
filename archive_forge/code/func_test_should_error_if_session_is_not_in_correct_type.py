from unittest import mock
import warnings
from oslotest import base
from monascaclient import client
@mock.patch('monascaclient.client.migration')
@mock.patch('monascaclient.client._get_auth_handler')
@mock.patch('monascaclient.client._get_session')
def test_should_error_if_session_is_not_in_correct_type(self, _, __, ___):
    api_version = mock.Mock()
    for cls in [str, int, float]:
        session = mock.Mock(spec=cls)
        self.assertRaises(RuntimeError, client.Client, api_version, session=session)