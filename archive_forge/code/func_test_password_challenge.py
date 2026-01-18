import base64
import sys
import mock
import pytest  # type: ignore
import pyu2f  # type: ignore
from google.auth import exceptions
from google.oauth2 import challenges
@mock.patch('getpass.getpass', return_value='foo')
def test_password_challenge(getpass_mock):
    challenge = challenges.PasswordChallenge()
    with mock.patch('getpass.getpass', return_value='foo'):
        assert challenge.is_locally_eligible
        assert challenge.name == 'PASSWORD'
        assert challenges.PasswordChallenge().obtain_challenge_input({}) == {'credential': 'foo'}
    with mock.patch('getpass.getpass', return_value=None):
        assert challenges.PasswordChallenge().obtain_challenge_input({}) == {'credential': ' '}