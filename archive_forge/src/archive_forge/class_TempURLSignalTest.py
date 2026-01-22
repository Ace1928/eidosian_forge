from unittest import mock
import swiftclient.client
import testscenarios
import testtools
from testtools import matchers
import time
from heatclient.common import deployment_utils
from heatclient import exc
from heatclient.v1 import software_configs
class TempURLSignalTest(testtools.TestCase):

    @mock.patch.object(swiftclient.client, 'Connection')
    def test_create_swift_client(self, sc_conn):
        auth = mock.MagicMock()
        auth.get_token.return_value = '1234'
        auth.get_endpoint.return_value = 'http://192.0.2.1:8080'
        session = mock.MagicMock()
        args = mock.MagicMock()
        args.os_region_name = 'Region1'
        args.os_project_name = 'project'
        args.os_username = 'user'
        args.os_cacert = None
        args.insecure = True
        sc_conn.return_value = mock.MagicMock()
        sc = deployment_utils.create_swift_client(auth, session, args)
        self.assertEqual(sc_conn.return_value, sc)
        self.assertEqual(mock.call(session), auth.get_token.call_args)
        self.assertEqual(mock.call(session, service_type='object-store', region_name='Region1'), auth.get_endpoint.call_args)
        self.assertEqual(mock.call(cacert=None, insecure=True, key=None, tenant_name='project', preauthtoken='1234', authurl=None, user='user', preauthurl='http://192.0.2.1:8080', auth_version='2.0'), sc_conn.call_args)

    def test_create_temp_url(self):
        swift_client = mock.MagicMock()
        swift_client.url = 'http://fake-host.com:8080/v1/AUTH_demo'
        swift_client.head_account = mock.Mock(return_value={'x-account-meta-temp-url-key': '123456'})
        swift_client.post_account = mock.Mock()
        uuid_pattern = '[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89aAbB][a-f0-9]{3}-[a-f0-9]{12}'
        url = deployment_utils.create_temp_url(swift_client, 'bar', 60)
        self.assertFalse(swift_client.post_account.called)
        regexp = 'http://fake-host.com:8080/v1/AUTH_demo/bar-%s/%s\\?temp_url_sig=[0-9a-f]{40,64}&temp_url_expires=[0-9]{10}' % (uuid_pattern, uuid_pattern)
        self.assertThat(url, matchers.MatchesRegex(regexp))
        timeout = int(url.split('=')[-1])
        self.assertTrue(timeout < time.time() + 2 * 365 * 24 * 60 * 60)

    def test_get_temp_url_no_account_key(self):
        swift_client = mock.MagicMock()
        swift_client.url = 'http://fake-host.com:8080/v1/AUTH_demo'
        head_account = {}

        def post_account(data):
            head_account.update(data)
        swift_client.head_account = mock.Mock(return_value=head_account)
        swift_client.post_account = post_account
        self.assertNotIn('x-account-meta-temp-url-key', head_account)
        deployment_utils.create_temp_url(swift_client, 'bar', 60, 'foo')
        self.assertIn('x-account-meta-temp-url-key', head_account)

    def test_build_signal_id_no_signal(self):
        hc = mock.MagicMock()
        args = mock.MagicMock()
        args.signal_transport = 'NO_SIGNAL'
        self.assertIsNone(deployment_utils.build_signal_id(hc, args))

    def test_build_signal_id_no_client_auth(self):
        hc = mock.MagicMock()
        args = mock.MagicMock()
        args.os_no_client_auth = True
        args.signal_transport = 'TEMP_URL_SIGNAL'
        e = self.assertRaises(exc.CommandError, deployment_utils.build_signal_id, hc, args)
        self.assertEqual('Cannot use --os-no-client-auth, auth required to create a Swift TempURL.', str(e))

    @mock.patch.object(deployment_utils, 'create_temp_url')
    @mock.patch.object(deployment_utils, 'create_swift_client')
    def test_build_signal_id(self, csc, ctu):
        hc = mock.MagicMock()
        args = mock.MagicMock()
        args.name = 'foo'
        args.timeout = 60
        args.os_no_client_auth = False
        args.signal_transport = 'TEMP_URL_SIGNAL'
        csc.return_value = mock.MagicMock()
        temp_url = 'http://fake-host.com:8080/v1/AUTH_demo/foo/a81a74d5-c395-4269-9670-ddd0824fd696?temp_url_sig=6a68371d602c7a14aaaa9e3b3a63b8b85bd9a503&temp_url_expires=1425270977'
        ctu.return_value = temp_url
        self.assertEqual(temp_url, deployment_utils.build_signal_id(hc, args))
        self.assertEqual(mock.call(hc.http_client.auth, hc.http_client.session, args), csc.call_args)
        self.assertEqual(mock.call(csc.return_value, 'foo', 60), ctu.call_args)