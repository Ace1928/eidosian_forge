from tests.compat import mock, unittest
class TestSSHTimeout(unittest.TestCase):

    @unittest.skipIf(not paramiko, 'Paramiko missing')
    def test_timeout(self):
        client_tmp = paramiko.SSHClient

        def client_mock():
            client = client_tmp()
            client.connect = mock.Mock(name='connect')
            return client
        paramiko.SSHClient = client_mock
        paramiko.RSAKey.from_private_key_file = mock.Mock()
        server = mock.Mock()
        test = SSHClient(server)
        self.assertEqual(test._ssh_client.connect.call_args[1]['timeout'], None)
        test2 = SSHClient(server, timeout=30)
        self.assertEqual(test2._ssh_client.connect.call_args[1]['timeout'], 30)