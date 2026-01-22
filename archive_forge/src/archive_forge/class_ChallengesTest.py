import base64
import json
import os
import unittest
import mock
from google_reauth import challenges, errors
import pyu2f
class ChallengesTest(unittest.TestCase):
    """This class contains tests for reauth challanges. """

    @mock.patch('pyu2f.u2f.GetLocalU2FInterface', return_value=_u2f_interface_mock)
    def testSecurityKeyError(self, u2f_mock):
        metadata = {'status': 'READY', 'challengeId': 2, 'challengeType': 'SECURITY_KEY', 'securityKey': {'applicationId': 'security_key_application_id', 'challenges': [{'keyHandle': 'some_key', 'challenge': base64.urlsafe_b64encode('some_challenge'.encode('ascii')).decode('ascii')}]}}
        challenge = challenges.SecurityKeyChallenge()
        _u2f_interface_mock.error = pyu2f.errors.U2FError(pyu2f.errors.U2FError.DEVICE_INELIGIBLE)
        self.assertEqual(None, challenge.obtain_challenge_input(metadata))
        _u2f_interface_mock.error = pyu2f.errors.U2FError(pyu2f.errors.U2FError.TIMEOUT)
        self.assertEqual(None, challenge.obtain_challenge_input(metadata))
        _u2f_interface_mock.error = pyu2f.errors.NoDeviceFoundError()
        self.assertEqual(None, challenge.obtain_challenge_input(metadata))
        _u2f_interface_mock.error = pyu2f.errors.U2FError(pyu2f.errors.U2FError.BAD_REQUEST)
        with self.assertRaises(pyu2f.errors.U2FError):
            challenge.obtain_challenge_input(metadata)
        _u2f_interface_mock.error = pyu2f.errors.UnsupportedVersionException()
        with self.assertRaises(pyu2f.errors.UnsupportedVersionException):
            challenge.obtain_challenge_input(metadata)

    @mock.patch('getpass.getpass', return_value=None)
    def testNoPassword(self, getpass_mock):
        self.assertEqual(challenges.PasswordChallenge().obtain_challenge_input({}), {'credential': ' '})

    def testSaml(self):
        metadata = {'status': 'READY', 'challengeId': 1, 'challengeType': 'SAML', 'securityKey': {}}
        challenge = challenges.SamlChallenge()
        self.assertEqual(True, challenge.is_locally_eligible)
        with self.assertRaises(errors.ReauthSamlLoginRequiredError):
            challenge.obtain_challenge_input(metadata)