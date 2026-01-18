import base64
import sys
import mock
import pytest  # type: ignore
import pyu2f  # type: ignore
from google.auth import exceptions
from google.oauth2 import challenges
def test_saml_challenge():
    challenge = challenges.SamlChallenge()
    assert challenge.is_locally_eligible
    assert challenge.name == 'SAML'
    with pytest.raises(exceptions.ReauthSamlChallengeFailError):
        challenge.obtain_challenge_input(None)