from unittest import mock
import warnings
from oslotest import base
from monascaclient import client
@mock.patch('monascaclient.client.migration')
@mock.patch('monascaclient.client._get_auth_handler')
@mock.patch('monascaclient.client._get_session')
def test_should_reuse_the_session_if_initialized_with_one(self, get_session, get_auth, _):
    from keystoneauth1 import session as k_session
    api_version = mock.Mock()
    session = mock.Mock(spec=k_session.Session)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        client.Client(api_version, session=session)
        self.assertEqual(0, len(w))
    get_auth.assert_not_called()
    get_session.assert_not_called()