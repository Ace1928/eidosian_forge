import testtools
from unittest import mock
from troveclient.compat import auth
from troveclient.compat import exceptions
class Auth1_1Test(testtools.TestCase):

    def test_authenticate(self):
        check_url_none(self, auth.Auth1_1)
        username = 'trove_user'
        password = 'trove_password'
        url = 'test_url'
        authObj = auth.Auth1_1(url=url, type=auth.Auth1_1, client=None, username=username, password=password, tenant=None)

        def side_effect_func(auth_url, body, root_key):
            return (auth_url, body, root_key)
        mock_obj = mock.Mock()
        mock_obj.side_effect = side_effect_func
        authObj._authenticate = mock_obj
        auth_url, body, root_key = authObj.authenticate()
        self.assertEqual(username, body['credentials']['username'])
        self.assertEqual(password, body['credentials']['key'])
        self.assertEqual(auth_url, url)
        self.assertEqual('auth', root_key)