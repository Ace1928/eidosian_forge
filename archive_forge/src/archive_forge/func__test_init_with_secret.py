import testtools
from unittest import mock
from keystoneauth1.exceptions import catalog
from magnumclient.v1 import client
def _test_init_with_secret(self, init_func, mock_load_service_type, mock_load_session, mock_http_client):
    expected_password = 'expected_password'
    session = mock.Mock()
    mock_load_session.return_value = session
    init_func(expected_password)
    load_session_args = self._load_session_kwargs()
    load_session_args['password'] = expected_password
    mock_load_session.assert_called_once_with(**load_session_args)
    mock_load_service_type.assert_called_once_with(session, **self._load_service_type_kwargs())
    mock_http_client.assert_called_once_with(**self._session_client_kwargs(session))