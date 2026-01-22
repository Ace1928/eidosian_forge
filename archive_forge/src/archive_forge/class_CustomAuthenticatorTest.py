import base64
import json
import struct
import sys
import mock
from pyu2f import errors
from pyu2f import model
from pyu2f.convenience import customauthenticator
@mock.patch.object(sys, 'stderr', new=mock.MagicMock())
class CustomAuthenticatorTest(unittest.TestCase):

    @mock.patch.object(customauthenticator.os.environ, 'get', return_value=None)
    def testEnvVarNotSet(self, os_get_method):
        authenticator = customauthenticator.CustomAuthenticator('testorigin')
        self.assertFalse(authenticator.IsAvailable(), 'Should return false when no env var is present')
        with self.assertRaises(errors.PluginError):
            authenticator.Authenticate('appid', {})

    @mock.patch.object(customauthenticator.subprocess, 'Popen')
    @mock.patch.object(customauthenticator.os.environ, 'get', return_value='gnubbyagent')
    def testSuccessAuthenticate(self, os_get_method, popen_method):
        """Test plugin Authenticate success."""
        valid_plugin_response = {'type': 'sign_helper_reply', 'code': 0, 'errorDetail': '', 'responseData': {'appIdHash': SIGN_SUCCESS['app_id_hash_encoded'], 'challengeHash': SIGN_SUCCESS['challenge_hash_encoded'], 'keyHandle': SIGN_SUCCESS['key_handle_encoded'], 'version': SIGN_SUCCESS['u2f_version'], 'signatureData': SIGN_SUCCESS['signature_data_encoded']}, 'data': None}
        valid_plugin_response_json = json.dumps(valid_plugin_response).encode()
        valid_plugin_response_len = struct.pack('<I', len(valid_plugin_response_json))
        mock_communicate_method = mock.MagicMock()
        mock_communicate_method.return_value = [valid_plugin_response_len + valid_plugin_response_json]
        mock_wait_method = mock.MagicMock()
        mock_wait_method.return_value = 0
        process_mock = mock.MagicMock()
        process_mock.communicate = mock_communicate_method
        process_mock.wait = mock_wait_method
        popen_method.return_value = process_mock
        authenticator = customauthenticator.CustomAuthenticator(SIGN_SUCCESS['origin'])
        self.assertTrue(authenticator.IsAvailable(), 'Should return true if env var is present')
        self.assertTrue(os_get_method.called)
        challenge_data = [{'key': SIGN_SUCCESS['registered_key'], 'challenge': SIGN_SUCCESS['challenge']}]
        result = authenticator.Authenticate(SIGN_SUCCESS['app_id'], challenge_data)
        self.assertTrue(popen_method.called)
        self.assertTrue(mock_communicate_method.called)
        communicate_args = mock_communicate_method.call_args[0]
        self.assertEquals(len(communicate_args), 1, 'communicate() should have been called with two args')
        communicate_stdin = communicate_args[0]
        communicate_json_len_le = communicate_stdin[:4]
        communicate_json_len = struct.unpack('<I', communicate_json_len_le)[0]
        communicate_json = communicate_stdin[4:]
        self.assertEquals(len(communicate_json), communicate_json_len, 'communicate() should have been called with correctlength field')
        communicate_dict = json.loads(communicate_json)
        self.assertEquals(communicate_dict.get('type'), 'sign_helper_request')
        self.assertEquals(communicate_dict.get('timeoutSeconds'), 5)
        self.assertEquals(communicate_dict.get('localAlways'), True)
        challenges = communicate_dict.get('signData')
        self.assertIsNotNone(challenges)
        self.assertEquals(len(challenges), 1)
        challenge = challenges[0]
        self.assertEquals(challenge.get('appIdHash'), SIGN_SUCCESS['app_id_hash_encoded'])
        self.assertEquals(challenge.get('challengeHash'), SIGN_SUCCESS['challenge_hash_encoded'])
        self.assertEquals(challenge.get('keyHandle'), SIGN_SUCCESS['key_handle_encoded'])
        self.assertEquals(challenge.get('version'), SIGN_SUCCESS['u2f_version'])
        mock_wait_method.assert_called_with()
        self.assertEquals(result['applicationId'], SIGN_SUCCESS['app_id'])
        self.assertEquals(result['clientData'], SIGN_SUCCESS['client_data_encoded'])
        self.assertEquals(result['keyHandle'], SIGN_SUCCESS['key_handle_encoded'])
        self.assertEquals(result['signatureData'], SIGN_SUCCESS['signature_data_encoded'])

    @mock.patch.object(customauthenticator.subprocess, 'Popen')
    @mock.patch.object(customauthenticator.os.environ, 'get', return_value='gnubbyagent')
    def testPluginReturnsMalformedJson(self, os_get_method, popen_method):
        """Test when plugin returns non-json response."""
        plugin_response = b'abc'
        plugin_response_len = struct.pack('<I', len(plugin_response))
        mock_communicate_method = mock.MagicMock()
        mock_communicate_method.return_value = [plugin_response_len + plugin_response]
        mock_wait_method = mock.MagicMock()
        mock_wait_method.return_value = 0
        process_mock = mock.MagicMock()
        process_mock.communicate = mock_communicate_method
        process_mock.wait = mock_wait_method
        popen_method.return_value = process_mock
        challenge_data = [{'key': SIGN_SUCCESS['registered_key'], 'challenge': SIGN_SUCCESS['challenge']}]
        authenticator = customauthenticator.CustomAuthenticator(SIGN_SUCCESS['origin'])
        with self.assertRaises(errors.PluginError):
            authenticator.Authenticate(SIGN_SUCCESS['app_id'], challenge_data)

    @mock.patch.object(customauthenticator.subprocess, 'Popen')
    @mock.patch.object(customauthenticator.os.environ, 'get', return_value='gnubbyagent')
    def testPluginResponseMissingType(self, os_get_method, popen_method):
        """Test when plugin response is missing sign_helper_reply type."""
        valid_plugin_response = {'errorDetail': '', 'code': 0, 'responseData': {'appIdHash': SIGN_SUCCESS['app_id_hash_encoded'], 'challengeHash': SIGN_SUCCESS['challenge_hash_encoded'], 'keyHandle': SIGN_SUCCESS['key_handle_encoded'], 'version': SIGN_SUCCESS['u2f_version'], 'signatureData': SIGN_SUCCESS['signature_data_encoded']}, 'data': None}
        plugin_response_json = json.dumps(valid_plugin_response).encode()
        plugin_response_len = struct.pack('<I', len(plugin_response_json))
        mock_communicate_method = mock.MagicMock()
        mock_communicate_method.return_value = [plugin_response_len + plugin_response_json]
        mock_wait_method = mock.MagicMock()
        mock_wait_method.return_value = 0
        process_mock = mock.MagicMock()
        process_mock.communicate = mock_communicate_method
        process_mock.wait = mock_wait_method
        popen_method.return_value = process_mock
        challenge_data = [{'key': SIGN_SUCCESS['registered_key'], 'challenge': SIGN_SUCCESS['challenge']}]
        authenticator = customauthenticator.CustomAuthenticator(SIGN_SUCCESS['origin'])
        with self.assertRaises(errors.PluginError):
            authenticator.Authenticate(SIGN_SUCCESS['app_id'], challenge_data)

    @mock.patch.object(customauthenticator.subprocess, 'Popen')
    @mock.patch.object(customauthenticator.os.environ, 'get', return_value='gnubbyagent')
    def testPluginReturnsIncompleteResponse(self, os_get_method, popen_method):
        """Test when plugin response has missing fields."""
        valid_plugin_response = {'type': 'sign_helper_reply', 'errorDetail': '', 'responseData': {'appIdHash': SIGN_SUCCESS['app_id_hash_encoded'], 'challengeHash': SIGN_SUCCESS['challenge_hash_encoded'], 'keyHandle': SIGN_SUCCESS['key_handle_encoded'], 'version': SIGN_SUCCESS['u2f_version'], 'signatureData': SIGN_SUCCESS['signature_data_encoded']}, 'data': None}
        plugin_response_json = json.dumps(valid_plugin_response).encode()
        plugin_response_len = struct.pack('<I', len(plugin_response_json))
        mock_communicate_method = mock.MagicMock()
        mock_communicate_method.return_value = [plugin_response_len + plugin_response_json]
        mock_wait_method = mock.MagicMock()
        mock_wait_method.return_value = 0
        process_mock = mock.MagicMock()
        process_mock.communicate = mock_communicate_method
        process_mock.wait = mock_wait_method
        popen_method.return_value = process_mock
        challenge_data = [{'key': SIGN_SUCCESS['registered_key'], 'challenge': SIGN_SUCCESS['challenge']}]
        authenticator = customauthenticator.CustomAuthenticator(SIGN_SUCCESS['origin'])
        with self.assertRaises(errors.PluginError):
            authenticator.Authenticate(SIGN_SUCCESS['app_id'], challenge_data)

    @mock.patch.object(customauthenticator.subprocess, 'Popen')
    @mock.patch.object(customauthenticator.os.environ, 'get', return_value='gnubbyagent')
    def testPluginReturnsTouchRequired(self, os_get_method, popen_method):
        """Test when plugin with error 'touch required'."""
        valid_plugin_response = {'type': 'sign_helper_reply', 'errorDetail': '', 'code': customauthenticator.SK_SIGNING_PLUGIN_TOUCH_REQUIRED, 'responseData': {'appIdHash': SIGN_SUCCESS['app_id_hash_encoded'], 'challengeHash': SIGN_SUCCESS['challenge_hash_encoded'], 'keyHandle': SIGN_SUCCESS['key_handle_encoded'], 'version': SIGN_SUCCESS['u2f_version'], 'signatureData': SIGN_SUCCESS['signature_data_encoded']}, 'data': None}
        plugin_response_json = json.dumps(valid_plugin_response).encode()
        plugin_response_len = struct.pack('<I', len(plugin_response_json))
        mock_communicate_method = mock.MagicMock()
        mock_communicate_method.return_value = [plugin_response_len + plugin_response_json]
        mock_wait_method = mock.MagicMock()
        mock_wait_method.return_value = 0
        process_mock = mock.MagicMock()
        process_mock.communicate = mock_communicate_method
        process_mock.wait = mock_wait_method
        popen_method.return_value = process_mock
        challenge_data = [{'key': SIGN_SUCCESS['registered_key'], 'challenge': SIGN_SUCCESS['challenge']}]
        authenticator = customauthenticator.CustomAuthenticator(SIGN_SUCCESS['origin'])
        with self.assertRaises(errors.U2FError) as cm:
            authenticator.Authenticate(SIGN_SUCCESS['app_id'], challenge_data)
        self.assertEquals(cm.exception.code, errors.U2FError.TIMEOUT)

    @mock.patch.object(customauthenticator.subprocess, 'Popen')
    @mock.patch.object(customauthenticator.os.environ, 'get', return_value='gnubbyagent')
    def testPluginReturnsWrongData(self, os_get_method, popen_method):
        """Test when plugin with error 'wrong data'."""
        valid_plugin_response = {'type': 'sign_helper_reply', 'errorDetail': '', 'code': customauthenticator.SK_SIGNING_PLUGIN_WRONG_DATA, 'responseData': {'appIdHash': SIGN_SUCCESS['app_id_hash_encoded'], 'challengeHash': SIGN_SUCCESS['challenge_hash_encoded'], 'keyHandle': SIGN_SUCCESS['key_handle_encoded'], 'version': SIGN_SUCCESS['u2f_version'], 'signatureData': SIGN_SUCCESS['signature_data_encoded']}, 'data': None}
        plugin_response_json = json.dumps(valid_plugin_response).encode()
        plugin_response_len = struct.pack('<I', len(plugin_response_json))
        mock_communicate_method = mock.MagicMock()
        mock_communicate_method.return_value = [plugin_response_len + plugin_response_json]
        mock_wait_method = mock.MagicMock()
        mock_wait_method.return_value = 0
        process_mock = mock.MagicMock()
        process_mock.communicate = mock_communicate_method
        process_mock.wait = mock_wait_method
        popen_method.return_value = process_mock
        challenge_data = [{'key': SIGN_SUCCESS['registered_key'], 'challenge': SIGN_SUCCESS['challenge']}]
        authenticator = customauthenticator.CustomAuthenticator(SIGN_SUCCESS['origin'])
        with self.assertRaises(errors.U2FError) as cm:
            authenticator.Authenticate(SIGN_SUCCESS['app_id'], challenge_data)
        self.assertEquals(cm.exception.code, errors.U2FError.DEVICE_INELIGIBLE)